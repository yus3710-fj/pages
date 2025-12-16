---
layout: post
title:  "FortranをMLIRで表現する話（最適化編）"
date:   2025-12-17 12:00:00 +0900
categories: コンパイラ
tag: llvm mlir
---
この記事は [FUJITSU Advent Calendar 2025](https://qiita.com/advent-calendar/2025/fujitsu) 17日目の記事です。  
なお、本記事は個人の意見に基づくものであり、組織を代表するものではありません。

## 概要
[去年](../../../2024/12/14/fortran_with_mlir.html#section)の記事と同じです。
去年はFIRを使わずMLIR付属のdialectのみに置き換えられるか？という検証で終わってしまい、肝心の最適化観点での検証が出来ていませんでした。
今年こそはやります。
(先行研究が存在しているので相変わらずn番煎じなのですが。)

ちなみに1年経って状況が変わっており、以前の記事は内容が古くなっている部分もあるので注意が必要です。
(ここで全てを列挙するのは~~面倒なので~~本筋から逸れるのでやりません。)

## 検証
今回対象とするプログラムはシンプルな完全ネストループのみを含むプログラムとします。
いろんな最適化パスを通してみて、実行時間がどのように変化するかを観測します。
(予防線を張っておくと、プログラムがシンプル過ぎて面白い結果は得られないと思います。)

* 検証環境
  * CPU: Intel® Core™ i7-9750H
    * 2.60GHz
    * 6C12T
  * RAM: 8GB (WSL 2のデフォルト上限値)
  * OS: Ubuntu 24.04 (WSL 2)

* テストプログラム
```fortran
subroutine func(a, b, c)
  implicit none
  integer :: i, j, k
  integer, parameter :: size = 400
  real :: a(size,size), b(size,size), c(size,size)

  do i = 1, size
    do j = 1, size
      a(j,i) = a(j,i) + b(j,i)
    end do
  end do
end subroutine
```

* 実行コマンド
  * LLVM IR生成(例)
    ```console
    $ mlir-opt -pass-pipeline="builtin.module(convert-func-to-llvm)" func.mlir | mlir-translate --mlir-to-llvmir -o func.ll
    ```
  * 実行バイナリ生成
    ```console
    $ flang -O3 -ffast-math -march=native main.f90 func.ll
    ```
  * 実行
    ```console
    $ time ./a.out
    
    real    0m0.010s # これを実行時間として採用
    user    0m0.002s
    sys     0m0.002s
    ```

### LEVEL0: Flangの純粋なパフォーマンス
まずはFIR周りの最適化とLLVMバックエンドの最適化でどこまでの性能が出せるのか確認しておきます。

Flangが生成するHLFIRは以下です。(`-Xflang -save-temps`オプションで得られます。)

```
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "skylake", fir.target_features = #llvm.target_features<["+prfchw", "-cldemote", "+avx", "+aes", "+sahf", "+pclmul", "-xop", "+crc32", "-amx-fp8", "+xsaves", "-avx512fp16", "-usermsr", "-sm4", "-egpr", "+sse4.1", "-avx10.1", "-avx512ifma", "+xsave", "+sse4.2", "-tsxldtrk", "-sm3", "-ptwrite", "-widekl", "-movrs", "+invpcid", "+64bit", "+xsavec", "-avx512vpopcntdq", "+cmov", "-avx512vp2intersect", "-avx512cd", "+movbe", "-avxvnniint8", "-ccmp", "-amx-int8", "-kl", "-sha512", "-avxvnni", "-rtm", "+adx", "+avx2", "-hreset", "-movdiri", "-serialize", "-vpclmulqdq", "-avx512vl", "-uintr", "-cf", "+clflushopt", "-raoint", "-cmpccxadd", "+bmi", "-amx-tile", "+sse", "-gfni", "-avxvnniint16", "-amx-fp16", "-zu", "-ndd", "+xsaveopt", "+rdrnd", "-avx512f", "-amx-bf16", "-avx512bf16", "-avx512vnni", "-push2pop2", "+cx8", "-avx512bw", "+sse3", "-pku", "-nf", "-amx-tf32", "-amx-avx512", "+fsgsbase", "-clzero", "-mwaitx", "-lwp", "+lzcnt", "-sha", "-movdir64b", "-ppx", "-wbnoinvd", "-enqcmd", "-amx-transpose", "-avxneconvert", "-tbm", "-pconfig", "-amx-complex", "+ssse3", "+cx16", "-avx10.2", "+bmi2", "+fma", "+popcnt", "-avxifma", "+f16c", "-avx512bitalg", "-rdpru", "-clwb", "+mmx", "+sse2", "+rdseed", "-avx512vbmi2", "-prefetchi", "-amx-movrs", "-rdpid", "-fma4", "-avx512vbmi", "-shstk", "-vaes", "-waitpkg", "-sgx", "+fxsr", "-avx512dq", "-sse4a"]>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (https://github.com/llvm/llvm-project.git ef46f8a7d73c1657b2448fc2f3f41ff6eecc4c0f)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPfunc(%arg0: !fir.ref<!fir.array<400x400xf32>> {fir.bindc_name = "a"}, %arg1: !fir.ref<!fir.array<400x400xf32>> {fir.bindc_name = "b"}, %arg2: !fir.ref<!fir.array<400x400xf32>> {fir.bindc_name = "c"}) {
    %0 = fir.dummy_scope : !fir.dscope
    %c400 = arith.constant 400 : index
    %c400_0 = arith.constant 400 : index
    %1 = fir.shape %c400, %c400_0 : (index, index) -> !fir.shape<2>
    %2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFfuncEa"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %c400_1 = arith.constant 400 : index
    %c400_2 = arith.constant 400 : index
    %3 = fir.shape %c400_1, %c400_2 : (index, index) -> !fir.shape<2>
    %4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFfuncEb"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %c400_3 = arith.constant 400 : index
    %c400_4 = arith.constant 400 : index
    %5 = fir.shape %c400_3, %c400_4 : (index, index) -> !fir.shape<2>
    %6:2 = hlfir.declare %arg2(%5) dummy_scope %0 {uniq_name = "_QFfuncEc"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %7 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFfuncEi"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFfuncEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9 = fir.alloca i32 {bindc_name = "j", uniq_name = "_QFfuncEj"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFfuncEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %11 = fir.alloca i32 {bindc_name = "k", uniq_name = "_QFfuncEk"}
    %12:2 = hlfir.declare %11 {uniq_name = "_QFfuncEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %13 = fir.address_of(@_QFfuncECsize) : !fir.ref<i32>
    %14:2 = hlfir.declare %13 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFfuncECsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %15 = fir.convert %c1_i32 : (i32) -> index
    %c400_i32 = arith.constant 400 : i32
    %16 = fir.convert %c400_i32 : (i32) -> index
    %c1 = arith.constant 1 : index
    %17 = fir.convert %15 : (index) -> i32
    %18 = fir.do_loop %arg3 = %15 to %16 step %c1 iter_args(%arg4 = %17) -> (i32) {
      fir.store %arg4 to %8#0 : !fir.ref<i32>
      %c1_i32_5 = arith.constant 1 : i32
      %19 = fir.convert %c1_i32_5 : (i32) -> index
      %c400_i32_6 = arith.constant 400 : i32
      %20 = fir.convert %c400_i32_6 : (i32) -> index
      %c1_7 = arith.constant 1 : index
      %21 = fir.convert %19 : (index) -> i32
      %22 = fir.do_loop %arg5 = %19 to %20 step %c1_7 iter_args(%arg6 = %21) -> (i32) {
        fir.store %arg6 to %10#0 : !fir.ref<i32>
        %26 = fir.load %10#0 : !fir.ref<i32>
        %27 = fir.convert %26 : (i32) -> i64
        %28 = fir.load %8#0 : !fir.ref<i32>
        %29 = fir.convert %28 : (i32) -> i64
        %30 = hlfir.designate %2#0 (%27, %29)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        %31 = fir.load %30 : !fir.ref<f32>
        %32 = fir.load %10#0 : !fir.ref<i32>
        %33 = fir.convert %32 : (i32) -> i64
        %34 = fir.load %8#0 : !fir.ref<i32>
        %35 = fir.convert %34 : (i32) -> i64
        %36 = hlfir.designate %4#0 (%33, %35)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        %37 = fir.load %36 : !fir.ref<f32>
        %38 = arith.addf %31, %37 fastmath<fast> : f32
        %39 = fir.load %10#0 : !fir.ref<i32>
        %40 = fir.convert %39 : (i32) -> i64
        %41 = fir.load %8#0 : !fir.ref<i32>
        %42 = fir.convert %41 : (i32) -> i64
        %43 = hlfir.designate %2#0 (%40, %42)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        hlfir.assign %38 to %43 : f32, !fir.ref<f32>
        %44 = fir.convert %c1_7 : (index) -> i32
        %45 = fir.load %10#0 : !fir.ref<i32>
        %46 = arith.addi %45, %44 overflow<nsw> : i32
        fir.result %46 : i32
      }
      fir.store %22 to %10#0 : !fir.ref<i32>
      %23 = fir.convert %c1 : (index) -> i32
      %24 = fir.load %8#0 : !fir.ref<i32>
      %25 = arith.addi %24, %23 overflow<nsw> : i32
      fir.result %25 : i32
    }
    fir.store %18 to %8#0 : !fir.ref<i32>
    return
  }
  fir.global internal @_QFfuncECsize constant : i32 {
    %c400_i32 = arith.constant 400 : i32
    fir.has_value %c400_i32 : i32
  }
}
```

LLVMバックエンドで適用されている最適化を`-Rpass=.*`オプションで確認すると、LICMとSIMD化とループ展開がかかっているようです。

```
func.f90:9:7: remark: hoisting zext [-Rpass=licm]
func.f90:9:7: remark: hoisting mul [-Rpass=licm]
func.f90:9:7: remark: hoisting add [-Rpass=licm]
func.f90:8:5: remark: vectorized loop (vectorization width: 8, interleaved count: 4) [-Rpass=loop-vectorize]
func.f90:8:5: remark: completely unrolled loop with 12 iterations [-Rpass=loop-unroll]
func.f90:8:5: remark: completely unrolled loop with 2 iterations [-Rpass=loop-unroll]
```

実行時間は以下のようになりました。

```
$ time ./a.out 

real    0m13.329s
user    0m13.576s
sys     0m0.001s
```

#### 解説
特になし

#### 考察
* 私の理解が正しければFlangフロントエンドでの最適化はCSEや正規化くらいしか適用されていないはずなので、バックエンドにほぼ頼り切りと言ってよいと思います

### LEVEL1: MLIRに変換するが、LLVMの最適化に丸投げ
LEVEL0どころかLEVEL-1な気もしますが、去年のおさらいです。

用意したMLIRは以下です。

```
#skylake_features = #llvm.target_features<["+prfchw", "-cldemote", "+avx", "+aes", "+sahf", "+pclmul", "-xop", "+crc32", "-amx-fp8", "+xsaves", "-avx512fp16", "-usermsr", "-sm4", "-egpr", "+sse4.1", "-avx10.1", "-avx512ifma", "+xsave", "+sse4.2", "-tsxldtrk", "-sm3", "-ptwrite", "-widekl", "-movrs", "+invpcid", "+64bit", "+xsavec", "-avx512vpopcntdq", "+cmov", "-avx512vp2intersect", "-avx512cd", "+movbe", "-avxvnniint8", "-ccmp", "-amx-int8", "-kl", "-sha512", "-avxvnni", "-rtm", "+adx", "+avx2", "-hreset", "-movdiri", "-serialize", "-vpclmulqdq", "-avx512vl", "-uintr", "-cf", "+clflushopt", "-raoint", "-cmpccxadd", "+bmi", "-amx-tile", "+sse", "-gfni", "-avxvnniint16", "-amx-fp16", "-zu", "-ndd", "+xsaveopt", "+rdrnd", "-avx512f", "-amx-bf16", "-avx512bf16", "-avx512vnni", "-push2pop2", "+cx8", "-avx512bw", "+sse3", "-pku", "-nf", "-amx-tf32", "-amx-avx512", "+fsgsbase", "-clzero", "-mwaitx", "-lwp", "+lzcnt", "-sha", "-movdir64b", "-ppx", "-wbnoinvd", "-enqcmd", "-amx-transpose", "-avxneconvert", "-tbm", "-pconfig", "-amx-complex", "+ssse3", "+cx16", "-avx10.2", "+bmi2", "+fma", "+popcnt", "-avxifma", "+f16c", "-avx512bitalg", "-rdpru", "-clwb", "+mmx", "+sse2", "+rdseed", "-avx512vbmi2", "-prefetchi", "-amx-movrs", "-rdpid", "-fma4", "-avx512vbmi", "-shstk", "-vaes", "-waitpkg", "-sgx", "+fxsr", "-avx512dq", "-sse4a"]>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (https://github.com/llvm/llvm-project.git ef46f8a7d73c1657b2448fc2f3f41ff6eecc4c0f)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @func_(%arg0: memref<400x400xf32> {fir.bindc_name = "a", llvm.noalias, llvm.nocapture}, %arg1: memref<400x400xf32> {fir.bindc_name = "b", llvm.noalias, llvm.nocapture}, %arg2: memref<400x400xf32> {fir.bindc_name = "c", llvm.noalias, llvm.nocapture}) attributes {fir.internal_name = "_QPfunc", no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, target_cpu = "skylake", target_features = #skylake_features, unsafe_fp_math = true} {
    //%0 = fir.dummy_scope : !fir.dscope
    %c400 = arith.constant 400 : index
    %c400_0 = arith.constant 400 : index
    %1 = shape.from_extents %c400, %c400_0 : index, index
    //%2:2 = hlfir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFfuncEa"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %c400_1 = arith.constant 400 : index
    %c400_2 = arith.constant 400 : index
    %3 = shape.from_extents %c400_1, %c400_2 : index, index
    //%4:2 = hlfir.declare %arg1(%3) dummy_scope %0 {uniq_name = "_QFfuncEb"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %c400_3 = arith.constant 400 : index
    %c400_4 = arith.constant 400 : index
    %5 = shape.from_extents %c400_3, %c400_4 : index, index
    //%6:2 = hlfir.declare %arg2(%5) dummy_scope %0 {uniq_name = "_QFfuncEc"} : (!fir.ref<!fir.array<400x400xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.ref<!fir.array<400x400xf32>>, !fir.ref<!fir.array<400x400xf32>>)
    %7 = memref.alloca() {bindc_name = "i", uniq_name = "_QFfuncEi"} : memref<i32>
    //%8:2 = hlfir.declare %7 {uniq_name = "_QFfuncEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %9 = memref.alloca() {bindc_name = "j", uniq_name = "_QFfuncEj"} : memref<i32>
    //%10:2 = hlfir.declare %9 {uniq_name = "_QFfuncEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %11 = memref.alloca() {bindc_name = "k", uniq_name = "_QFfuncEk"} : memref<i32>
    //%12:2 = hlfir.declare %11 {uniq_name = "_QFfuncEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %13 = memref.get_global @_QFfuncECsize : memref<i32>
    //%14:2 = hlfir.declare %13 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFfuncECsize"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1_i32 = arith.constant 1 : i32
    %15 = arith.index_cast %c1_i32 : i32 to index
    %c400_i32 = arith.constant 400 : i32
    %16 = arith.index_cast %c400_i32 : i32 to index
    %c1 = arith.constant 1 : index
    %17 = arith.index_cast %15 : index to i32
    %c0 = arith.constant 0 : index
    %dist_i = arith.subi %16, %15 : index
    %diff_i = arith.addi %dist_i, %c1 : index
    %tripcount_i = arith.divsi %diff_i, %c1 : index
    %18 = scf.for %arg3 = %c0 to %tripcount_i step %c1 iter_args(%arg4 = %17) -> (i32) {
      memref.store %arg4, %7[] : memref<i32>
      %c1_i32_5 = arith.constant 1 : i32
      %19 = arith.index_cast %c1_i32_5 : i32 to index
      %c400_i32_6 = arith.constant 400 : i32
      %20 = arith.index_cast %c400_i32_6 : i32 to index
      %c1_7 = arith.constant 1 : index
      %21 = arith.index_cast %19 : index to i32
      %c0_0 = arith.constant 0 : index
      %dist_j = arith.subi %20, %19 : index
      %diff_j = arith.addi %dist_i, %c1_7 : index
      %tripcount_j = arith.divsi %diff_i, %c1_7 : index
      %22 = scf.for %arg5 = %c0_0 to %tripcount_j step %c1_7 iter_args(%arg6 = %21) -> (i32) {
        memref.store %arg6, %9[] : memref<i32>
        %c-1 = arith.constant -1 : index
        %26 = memref.load %9[] : memref<i32>
        %27 = arith.index_cast %26 : i32 to index
        %28 = memref.load %7[] : memref<i32>
        %29 = arith.index_cast %28 : i32 to index
        //%30 = hlfir.designate %2#0 (%27, %29)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        %idx1 = arith.subi %27, %c1 overflow<nsw> : index
        %idx2 = arith.subi %29, %c1 overflow<nsw> : index
        %31 = memref.load %arg0[%idx2, %idx1] : memref<400x400xf32>
        %32 = memref.load %9[] : memref<i32>
        %33 = arith.index_cast %32 : i32 to index
        %34 = memref.load %7[] : memref<i32>
        %35 = arith.index_cast %34 : i32 to index
        //%36 = hlfir.designate %4#0 (%33, %35)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        %idx3 = arith.subi %33, %c1 overflow<nsw> : index
        %idx4 = arith.subi %35, %c1 overflow<nsw> : index
        %37 = memref.load %arg1[%idx4, %idx3] : memref<400x400xf32>
        %38 = arith.addf %31, %37 fastmath<fast> : f32
        %39 = memref.load %9[] : memref<i32>
        %40 = arith.index_cast %39 : i32 to index
        %41 = memref.load %7[] : memref<i32>
        %42 = arith.index_cast %41 : i32 to index
        //%43 = hlfir.designate %2#0 (%40, %42)  : (!fir.ref<!fir.array<400x400xf32>>, i64, i64) -> !fir.ref<f32>
        %idx5 = arith.subi %40, %c1 overflow<nsw> : index
        %idx6 = arith.subi %42, %c1 overflow<nsw> : index
        memref.store %38, %arg0[%idx6, %idx5] : memref<400x400xf32>
        %44 = arith.index_cast %c1_7 : index to i32
        %45 = memref.load %9[] : memref<i32>
        %46 = arith.addi %45, %44 overflow<nsw> : i32
        scf.yield %46 : i32
      }
      memref.store %22, %9[] : memref<i32>
      %23 = arith.index_cast %c1 : index to i32
      %24 = memref.load %7[] : memref<i32>
      %25 = arith.addi %24, %23 overflow<nsw> : i32
      scf.yield %25 : i32
    }
    memref.store %18, %7[] : memref<i32>
    return
  }
  memref.global "private" @_QFfuncECsize : memref<i32> = dense<400>
}
```

これを以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,canonicalize,cse,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" level1.mlir | mlir-translate --mlir-to-llvmir -o level1.ll
```

LLVMバックエンドで適用されている最適化を`-Rpass=.*`オプションで確認すると、先ほどと全く同じ最適化がかかっているようです。

```
remark: hoisting zext [-Rpass=licm]
remark: hoisting mul [-Rpass=licm]
remark: hoisting add [-Rpass=licm]
remark: vectorized loop (vectorization width: 8, interleaved count: 4) [-Rpass=loop-vectorize]
remark: completely unrolled loop with 12 iterations [-Rpass=loop-unroll]
remark: completely unrolled loop with 2 iterations [-Rpass=loop-unroll]
```

実行時間は以下のようになりました。

```
real    0m13.031s
user    0m13.549s
sys     0m0.000s
```

#### 解説
* `hlfir.declare`や`fir.dummy_scope`は無視
* `fir.alloca`は`memref.alloca`に
* グローバル変数は`memref.global`で定義し、`memref.get_global`でアドレスを取得
  * `internal`から`private`に変わっちゃいますが
* `fir.do_loop`は`scf.for`に
  * 最初に回転数を計算し、その分だけ回転
* `fir.shape`は`shape.from_extents`に
* `hlfir.designate`は消して、添字計算の処理を直接書きます
  * [去年](../../../2024/12/14/fortran_with_mlir.html#ver)は`memref.subview`を使っていましたが、実は`memref.subview`のoffsetがマイナスになってはいけないという制約があるらしく、それを静的に解析できるときはエラーになるので使えません
    * 元々部分配列用の機能なのでマイナスになってはいけないというのは、それはそう
    * ただ、去年のIRも静的に解析できそうなのに今もエラーにならないのはよく分かりません
* 引数の`memref`は`finalize-memref-to-llvm`の直後に`!llvm.ptr`に手で変換します
  * 理由は後述
* `fir.target_cpu`と`fir.target_features`は、値をそのままmodule内の各functionのattributeとして渡してあげればよいです
* Flangでは後段でTBAAというmetadataを付与しますが、面倒なので付けていません
  * 代わりに引数の`noalias` attributeがあれば、今回は問題なくエイリアス解析できるはず

#### 考察
* 去年は配列(というか`memref`)を引数として渡していなかったので気づかなかったですが、`!fir.ref`を`memref`に置き換えてはいけないようです
  * `convert-func-to-llvm`は関数の引数にある`memref`を分解してしまうため、`use-bare-ptr-memref-call-conv`を設定する必要がありますが、これでも渡ってくるのはDescriptorであって配列そのものではないため、結局手で直す必要が出てきます
    * 動けばOKということであればcaller側でDescriptorを渡すよう直してもいいのですが、余計なメモリアクセスが増えるので、性能を考えるとやはりDescriptorを介したアクセスは避けたいです
  * FlangのDescriptorとしては`!fir.box`があるので、引数の`memref`はFlang独自のパスで`!llvm.ptr`に置き換える、という運用が妥当かもしれません
    * ptr dialectが追加されているのですが、`ptr.load`/`store`がなぜか使えないので諦めました
    * そうなるともう`!fir.ref<!fir.array<...>>`のままでいいのでは？という気もしてきますね…
* 性能差については、アセンブリコードのdiffをとってもファイル名以外の差分はなかったので、理論上ないはずです(面白味には欠けますが)

### LEVEL2: Mem2Reg, LICM
やれる最適化はやり尽くしてしまっている感じがあるので、LLVMから仕事を奪っていこうと思います。
パッと目に付くものだとMem2RegとLICMがあるので、これをMLIRでやってしまいます。

以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(canonicalize,cse,mem2reg,loop-invariant-code-motion,func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,canonicalize,cse,convert-func-to-llvm,convert-cf-to-llvm,reconcile-unrealized-casts)" level1.mlir | mlir-translate --mlir-to-llvmir -o level2.ll
```

実行時間は以下のようになりました。

```
real    0m13.235s
user    0m13.517s
sys     0m0.014s
```

#### 解説
* Mem2Reg
  * ループの中でループ変数の値を逐一ストアしていますが、逐一メモリに書き出さなくても、レジスタに保持しておいてループが終わったらストアしても別にいいはずです。これをやってくれるのがMem2Regです。
* LICM(Loop Invariant Code Motion、ループ不変式の移動)
  * ループの中の処理で毎回同じ結果になるもの(例えば定数のロード)は、ループの直前に実行してループの中ではその結果を使い回した方が効率的です。

#### 考察
* コード変形がかかっていないらしく、そもそもLLVM IRに落とした時点で差分がありませんでした
  * バックエンドにあるものをわざわざフロントエンドでもやる意味は薄いので、まだ試験的な実装に留まっていそうです。


### LEVEL3: 自動並列化
プログラムの高速化の基本はループを速くすることですが、残念ながら`scf.for`だとやれることがほぼありません。
ということで半分ズルですが、どう見ても依存がないループなので並列実行します。
(GPUに投げてもいいですが、今回はオーバーヘッドの方が大きくなりそうなのでCPUでスレッド並列実行します。)

用意したMLIRは以下です。

```
#skylake_features = #llvm.target_features<["+prfchw", "-cldemote", "+avx", "+aes", "+sahf", "+pclmul", "-xop", "+crc32", "-amx-fp8", "+xsaves", "-avx512fp16", "-usermsr", "-sm4", "-egpr", "+sse4.1", "-avx10.1", "-avx512ifma", "+xsave", "+sse4.2", "-tsxldtrk", "-sm3", "-ptwrite", "-widekl", "-movrs", "+invpcid", "+64bit", "+xsavec", "-avx512vpopcntdq", "+cmov", "-avx512vp2intersect", "-avx512cd", "+movbe", "-avxvnniint8", "-ccmp", "-amx-int8", "-kl", "-sha512", "-avxvnni", "-rtm", "+adx", "+avx2", "-hreset", "-movdiri", "-serialize", "-vpclmulqdq", "-avx512vl", "-uintr", "-cf", "+clflushopt", "-raoint", "-cmpccxadd", "+bmi", "-amx-tile", "+sse", "-gfni", "-avxvnniint16", "-amx-fp16", "-zu", "-ndd", "+xsaveopt", "+rdrnd", "-avx512f", "-amx-bf16", "-avx512bf16", "-avx512vnni", "-push2pop2", "+cx8", "-avx512bw", "+sse3", "-pku", "-nf", "-amx-tf32", "-amx-avx512", "+fsgsbase", "-clzero", "-mwaitx", "-lwp", "+lzcnt", "-sha", "-movdir64b", "-ppx", "-wbnoinvd", "-enqcmd", "-amx-transpose", "-avxneconvert", "-tbm", "-pconfig", "-amx-complex", "+ssse3", "+cx16", "-avx10.2", "+bmi2", "+fma", "+popcnt", "-avxifma", "+f16c", "-avx512bitalg", "-rdpru", "-clwb", "+mmx", "+sse2", "+rdseed", "-avx512vbmi2", "-prefetchi", "-amx-movrs", "-rdpid", "-fma4", "-avx512vbmi", "-shstk", "-vaes", "-waitpkg", "-sgx", "+fxsr", "-avx512dq", "-sse4a"]>
#array_access = affine_map<(d0)[s0, s1] -> (d0 * s1 - s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (https://github.com/llvm/llvm-project.git ef46f8a7d73c1657b2448fc2f3f41ff6eecc4c0f)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @func_(%arg0: memref<400x400xf32> {fir.bindc_name = "a", llvm.noalias, llvm.nocapture}, %arg1: memref<400x400xf32> {fir.bindc_name = "b", llvm.noalias, llvm.nocapture}, %arg2: memref<400x400xf32> {fir.bindc_name = "c", llvm.noalias, llvm.nocapture}) attributes {fir.internal_name = "_QPfunc", no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, target_cpu = "skylake", target_features = #skylake_features, unsafe_fp_math = true} {
    %c400 = arith.constant 400 : index
    %c400_0 = arith.constant 400 : index
    %1 = shape.from_extents %c400, %c400_0 : index, index
    %c400_1 = arith.constant 400 : index
    %c400_2 = arith.constant 400 : index
    %3 = shape.from_extents %c400_1, %c400_2 : index, index
    %c400_3 = arith.constant 400 : index
    %c400_4 = arith.constant 400 : index
    %5 = shape.from_extents %c400_3, %c400_4 : index, index
    %7 = memref.alloca() {bindc_name = "i", uniq_name = "_QFfuncEi"} : memref<i32>
    %9 = memref.alloca() {bindc_name = "j", uniq_name = "_QFfuncEj"} : memref<i32>
    %11 = memref.alloca() {bindc_name = "k", uniq_name = "_QFfuncEk"} : memref<i32>
    %13 = memref.get_global @_QFfuncECsize : memref<i32>
    %c1_i32 = arith.constant 1 : i32
    %15 = arith.index_cast %c1_i32 : i32 to index
    %c400_i32 = arith.constant 400 : i32
    %16 = arith.index_cast %c400_i32 : i32 to index
    %c1 = arith.constant 1 : index
    %17 = arith.index_cast %15 : index to i32
    %c0 = arith.constant 0 : index
    %after_i = arith.addi %16, %c1 : index
    %18 = affine.for %arg3 = %15 to %after_i iter_args(%arg4 = %17) -> (i32) {
      affine.store %arg4, %7[] : memref<i32>
      %c1_i32_5 = arith.constant 1 : i32
      %19 = arith.index_cast %c1_i32_5 : i32 to index
      %c1_7 = arith.constant 1 : index
      %21 = arith.index_cast %19 : index to i32
      %22 = affine.for %arg5 = %15 to %after_i iter_args(%arg6 = %21) -> (i32) {
        affine.store %arg6, %9[] : memref<i32>
        %idx1 = affine.apply #array_access (%arg5)[%c1, %c1]
        %idx2 = affine.apply #array_access (%arg3)[%c1, %c1]
        %31 = affine.load %arg0[%idx2, %idx1] : memref<400x400xf32>
        %idx3 = affine.apply #array_access (%arg5)[%c1, %c1]
        %idx4 = affine.apply #array_access (%arg3)[%c1, %c1]
        %37 = affine.load %arg1[%idx4, %idx3] : memref<400x400xf32>
        %38 = arith.addf %31, %37 fastmath<fast> : f32
        %idx5 = affine.apply #array_access (%arg5)[%c1, %c1]
        %idx6 = affine.apply #array_access (%arg3)[%c1, %c1]
        affine.store %38, %arg0[%idx6, %idx5] : memref<400x400xf32>
        %44 = arith.index_cast %c1_7 : index to i32
        %45 = affine.load %9[] : memref<i32>
        %46 = arith.addi %45, %44 overflow<nsw> : i32
        affine.yield %46 : i32
      }
      affine.store %22, %9[] : memref<i32>
      %23 = arith.index_cast %c1 : index to i32
      %24 = affine.load %7[] : memref<i32>
      %25 = arith.addi %24, %23 overflow<nsw> : i32
      affine.yield %25 : i32
    }
    affine.store %18, %7[] : memref<i32>
    return
  }
  memref.global "private" @_QFfuncECsize : memref<i32> = dense<400>
}
```

これを以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(func.func(affine-loop-normalize,affine-scalrep,affine-parallelize{max-nested=1 parallel-reductions=true}),cse,func.func(lower-affine),convert-scf-to-openmp,canonicalize,finalize-memref-to-llvm,func.func(convert-scf-to-cf,convert-arith-to-llvm),canonicalize,cse,convert-func-to-llvm,convert-cf-to-llvm,convert-openmp-to-llvm,reconcile-unrealized-casts)" level1.mlir | mlir-translate --mlir-to-llvmir -o level3.ll
```

LLVMバックエンドで適用されている最適化を`-Rpass=.*`オプションで確認すると、LICMとGVNとSIMD化とループ展開がかかっているようです。

```
remark: hoisting mul [-Rpass=licm]
remark: load of type i32 eliminated [-Rpass=gvn]
remark: vectorized loop (vectorization width: 8, interleaved count: 4) [-Rpass=loop-vectorize]
remark: completely unrolled loop with 12 iterations [-Rpass=loop-unroll]
remark: completely unrolled loop with 2 iterations [-Rpass=loop-unroll]
remark: unrolled loop by a factor of 5 [-Rpass=loop-unroll]
remark: hoisting icmp [-Rpass=licm]
remark: hoisting icmp [-Rpass=licm]
remark: hoisting and [-Rpass=licm]
```

実行時間は以下のようになりました。(スレッド数は何も指定していないので12のはず)

```
real    0m6.899s
user    1m21.507s
sys     0m1.629s
```

#### 解説
* 前回の`affine.for`の使い方が言語規格違反だったので訂正しておきます
  * DO変数はループ実行完了後にその値を保持していないといけませんが、`affine.for`のループ誘導変数は保持されません。前回は`iter_args`にループ回転数を渡していましたが、ループ回転数は保持しておく必要がないので`iter_args`に渡す意味はありません。そのため、(違和感はすごいですが)`iter_args`にもループ誘導変数を持たせ、ループボディの末尾でインクリメントするコードを挿入しておき、AffineLoopNormalizeでループの回転数を計算させた方がおそらく良いです。
    * ただこの場合、affine dialectのベクトル化パスは動かせません。LLVM側でなんとかしてもらいましょう。うまくいかない可能性もありますが…([参考](https://github.com/llvm/llvm-project/pull/160630#issuecomment-3335028239))

#### 考察
* 内側ループも`affine.parallel`になれるのですが、そうすると二重に並列化されてしまいオーバーヘッドが大きくなってしまうため、最外側だけ変換させています。ただ、並列実行可能という情報は持たせておいて損はない気がするので、omp dialectへの変換時にもう少しうまくやってくれないかなという気持ちがあります。
  * というか二重の`affine.parallel`ではなく`affine.parallel`に2つのループインデックスの組を渡すようにするのがよさそうです。が、完全ネストの`affine.parallel`をまとめてくれるようなパスはあるのでしょうか？
    * よく見たら`affine.parallel`への変換パスの説明に「`affine.for`たちを1-Dの`affine.parallel`に変換する」と書いてありました。
    * リダクション変数があるのがダメかと思いましたが、消してもやはり二重の`affine.parallel`にしかなりませんでした。もしかしてそもそも機能として存在していないのでは…？
  * ちなみにFortranにおいてもFORALLやDO CONCURRENTはループインデックスの組を指定できるようになっていて、このとき各ループインデックスの値は(当然ですが)保存する必要がないです。
* 今回affineループであることは教えましたが、並列化できることの判定やOpenMPディレクティブの挿入をちゃんと自動でやってくれたのには驚きました。さすがに複雑なループになってくると解析精度は落ちるでしょうが、自明なループくらいコンパイラで並列化して欲しいという要望なら~~割と応えてくれそうという期待が持てました~~後述しますがちょっと怪しいです。

### LEVEL-EX: ループ交換
以下のようにループインデックスを逆に書いたとします。
これをlinalg dialectのループ交換を適用することで是正できるかを確認します。

```fortran
subroutine func(a, b, c)
  implicit none
  integer :: i, j, k
  integer, parameter :: size = 400
  real :: a(size,size), b(size,size), c(size,size)

  do i = 1, size
    do j = 1, size
      a(i,j) = a(i,j) + b(i,j)
    end do
  end do
end subroutine
```

用意したMLIRは以下です。

```
#skylake_features = #llvm.target_features<["+prfchw", "-cldemote", "+avx", "+aes", "+sahf", "+pclmul", "-xop", "+crc32", "-amx-fp8", "+xsaves", "-avx512fp16", "-usermsr", "-sm4", "-egpr", "+sse4.1", "-avx10.1", "-avx512ifma", "+xsave", "+sse4.2", "-tsxldtrk", "-sm3", "-ptwrite", "-widekl", "-movrs", "+invpcid", "+64bit", "+xsavec", "-avx512vpopcntdq", "+cmov", "-avx512vp2intersect", "-avx512cd", "+movbe", "-avxvnniint8", "-ccmp", "-amx-int8", "-kl", "-sha512", "-avxvnni", "-rtm", "+adx", "+avx2", "-hreset", "-movdiri", "-serialize", "-vpclmulqdq", "-avx512vl", "-uintr", "-cf", "+clflushopt", "-raoint", "-cmpccxadd", "+bmi", "-amx-tile", "+sse", "-gfni", "-avxvnniint16", "-amx-fp16", "-zu", "-ndd", "+xsaveopt", "+rdrnd", "-avx512f", "-amx-bf16", "-avx512bf16", "-avx512vnni", "-push2pop2", "+cx8", "-avx512bw", "+sse3", "-pku", "-nf", "-amx-tf32", "-amx-avx512", "+fsgsbase", "-clzero", "-mwaitx", "-lwp", "+lzcnt", "-sha", "-movdir64b", "-ppx", "-wbnoinvd", "-enqcmd", "-amx-transpose", "-avxneconvert", "-tbm", "-pconfig", "-amx-complex", "+ssse3", "+cx16", "-avx10.2", "+bmi2", "+fma", "+popcnt", "-avxifma", "+f16c", "-avx512bitalg", "-rdpru", "-clwb", "+mmx", "+sse2", "+rdseed", "-avx512vbmi2", "-prefetchi", "-amx-movrs", "-rdpid", "-fma4", "-avx512vbmi", "-shstk", "-vaes", "-waitpkg", "-sgx", "+fxsr", "-avx512dq", "-sse4a"]>
#cont_access = affine_map<(i, j) -> (i, j)>
#intrchg_access = affine_map<(i, j) -> (j, i)>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 22.0.0 (https://github.com/llvm/llvm-project.git ef46f8a7d73c1657b2448fc2f3f41ff6eecc4c0f)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @func_(%arg0: memref<400x400xf32> {fir.bindc_name = "a", llvm.noalias, llvm.nocapture}, %arg1: memref<400x400xf32> {fir.bindc_name = "b", llvm.noalias, llvm.nocapture}, %arg2: memref<400x400xf32> {fir.bindc_name = "c", llvm.noalias, llvm.nocapture}) attributes {fir.internal_name = "_QPfunc", no_infs_fp_math = true, no_nans_fp_math = true, no_signed_zeros_fp_math = true, target_cpu = "skylake", target_features = #skylake_features, unsafe_fp_math = true} {
    linalg.generic {indexing_maps = [#intrchg_access, #intrchg_access, #intrchg_access], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg1: memref<400x400xf32>, memref<400x400xf32>)
    outs(%arg0: memref<400x400xf32>) {
    ^bb0(%a_in: f32, %b: f32, %a_out: f32):
      %sum = arith.addf %a_in, %b : f32
      linalg.yield %sum : f32
    }
    return
  }
  memref.global "private" @_QFfuncECsize : memref<i32> = dense<400>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match ops{["linalg.generic"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    transform.structured.interchange %0 iterator_interchange = [1, 0] : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
```

これを以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(transform-interpreter,convert-linalg-to-parallel-loops,convert-scf-to-openmp,canonicalize,finalize-memref-to-llvm,func.func(convert-scf-to-cf,convert-arith-to-llvm),canonicalize,cse,convert-func-to-llvm,convert-cf-to-llvm,convert-openmp-to-llvm,reconcile-unrealized-casts)" level_ex1.mlir | mlir-translate --mlir-to-llvmir -o level_ex1.ll
```

ループ交換の有無での実行時間の違いは以下の通りです。

* ループ交換なし

```
real    2m59.802s
user    34m31.518s
sys     1m10.105s
```

* ループ交換あり

```
real    0m5.221s
user    1m0.951s
sys     0m1.297s
```

#### 解説
* linalg dialectは複雑なので掻い摘んで説明するに留めます（説明が間違っている可能性も高いのでその時は[ご指摘](https://github.com/yus3710-fj/pages/issues/new)ください）
  * 基本的に`linalg.generic`を使います。それ以外の具体的なOperationはLinalgMorphOpsPassというのを使って変換できるので、積極的に使う必要はなさそうです。
    * `in`と`out`にはそれぞれこのOperationの入出力の配列(定義済みのValue)を指定します。
    * attributeに指定が必要なものとして、`indexing_maps`と`iterator_types`があります。
      * `indexing_maps`は各入出力の配列の添字を指定します。もう少しちゃんと言うと、各ループインデックスと配列の各次元の添字式の対応関係を`affine_map`で指定します。
      * `iterator_types`は各ループの特性を外側ループから順に指定します。特性は`parallel`、`reduction`、`window`(?)の3つがあります。
  * linalg dialect単体の最適化パスはあまりなさそうに見えます。適用すべき最適化が分かっている場合は、transform dialectを使って明示的に指定すべきということです。
    * Halideのような感じで、アルゴリズム(計算式)をlinalg dialectで、スケジューリング(計算方法・手段)をtransform dialectで記述するのだと思います。
      * スケジューリングは`transform.with_named_sequence`というattributeを持つmoduleの中で、`@__transform_main`を定義することで記述できます。
    * ループ交換に対応するOperationは`transform.structured.interchange`です。
    * 現状ではtransform dialectを消してくれるパスがtest用(TestTransformDialectEraseSchedulePass)にしか実装されていません。しかもmoduleは消えてくれないので`mlir-translate`がうまく動きません。ということで引数の`memref`を`!llvm.ptr`に修正するタイミングで手動で消します。
      * スケジューリングは別のmoduleに書かないといけないわけではないので、元のmoduleにattributeを付けて`@__transform_main`を定義してもいいですが、個人的に好かないのでやっていません。
      * スケジューリングを別のファイルに記述しておいて、`mlir-transform-opt`というコマンドを使って適用するという方法もあるっぽいですが、Flangでそれをやるのは微妙かと思って試していません。

#### 考察
* 今回はループ交換の実装を適用したかったので無理やりlinalg dialectを使いましたが、ループとその中での各配列要素に対する処理をlinalg dialectで表現するのは適切ではなさそうです
  * どちらかというと配列記述(e.g., `a(:) = a(:) + b(:)`)などのelementalな処理を表現するのに使うのが正しいと思います。
  * この書き方だとDO変数の値を保存できないというのもあります。
  * ちなみにHLFIR dialectの実装の際、linalg dialectを使わない理由として、以下の2点が挙げられていました。(いずれも最適化観点ではなく、規格適合の観点)
    * `memref`/`tensor`を使わないと効果が薄い
    * `linalg.generic`で表現できる配列記述は限定的
* linalg dialect固有の変換としてループ交換が実装されているわけですが、使われているのがmatmulの変換と先述のtransform dialectくらいしかなさそうで、それならlinalgに閉じなくてもよくない？と思わなくもないです
  * まあ結局のところ、ループ交換が適用できるかの判断が難しいのかもしれません。(だから自明なパターンかユーザが指定した場合しか適用しない)
* 前のループ並列化の時と違い、今回は`scf.parallel`にループインデックスの組を渡せるため、特に何も考えずomp dialectへ変換できます
  * 出力されるLLVM IRを比較すると、ループインデックスの組を渡すとループが一重化されるようになります。
    * あとはリダクション変数の処理も消えてますが、(繰り返しになりますが)これはない方が問題です。
* ところで`perf stat`の結果を見る感じ、速度差の原因はキャッシュミスという感じでもなさそうに見えます
  * キャッシュミスの回数自体は増えていますがミス率で言うとほぼ変わっておらず、単純に実行命令数が増えているのが影響してそうです。(IPCも悪化してますが)
    * おそらく連続アクセスにならないことで諸々の最適化がかかりにくくなっているのでしょう。(このあたり調べる余裕がありませんでした…)

## 関連研究
Flangから見える取り組みをいくつか紹介します。

* `fir.do_loop` -> `scf.for` -> `affine.for`と変換し、種々の最適化を適用
  * 中国の企業が取り組んでいますが、Flangコミュニティからの関心をあまり惹けず、MLIRでの取り組みも苦戦しているように見えます。
    * また、明らかにFortranをMLIRに変換する部分の設計における考慮が不十分に見えます。
    * ただ、今回の記事を書くにあたり、参考にさせてもらった部分は多いです。
  * 個人的に`scf.for` -> `affine.for`の流れがよく分かりません。`fir.do_loop` -> `affine.for`で良いのでは？
  * と思ってたら最近になってNVIDIAも割と真面目に[検討している](https://github.com/llvm/llvm-project/pull/168703#issuecomment-3602386535)ことが判明しました。
  * あとこれまでの私の検証では`fir.dummy_scope`などは消してしまっていましたが、実際には必要な情報であり、しかもこれらのせいで最適化が阻害されることがあるということが[言及](https://discourse.llvm.org/t/does-memalloc-effect-allow-reordering/89136/3)されています。
* DO CONCURRENTの自動並列化
  * 言語仕様として、並列実行してよいことになっているため、それを活用した最適化の事例はFlangに限らず色々あります。
  * FlangではAMD主導でOpenMPのディレクティブの自動挿入を実装しているようです。
    * DO CONCURRENTの仕様自体がOpenMPに摺り寄ってきているので、筋がいいと言えるかもしれません。
* OpenMP 5.0の`loop`構文
  * 上と似ていますが、このディレクティブが付いたループも並列実行してよいことになります。
    * 実際にどのようなコード変形をするかは処理系に委ねられます。並列化するかもしれないし、SIMD化するかもしれません。あるいはGPUにオフロードするかもしれません。  
    (正直使い方がよく分かっていない…いい感じの解説記事も見つからないし)
  * これもAMD主導でFlangとMLIRのomp dialectへの実装が進められています。

今は最適化機会を如何に増やすかということよりも、ユーザからのヒント情報を確実に活かす仕組みを入れることがホットトピックになっている印象です。

## おわりに
今回の検証ではMLIR付属のdialectを使うことで最適化の幅が広がることが分かりました。
一方で既にLLVMでやっている最適化をMLIRで先行してやる意味は薄そうだというのも今回感じました。

個人的に面白そうな話ではある一方、Flangでこれを頑張る労力に見合う効果が得られるかはちょっと怪しいなと思いました。
