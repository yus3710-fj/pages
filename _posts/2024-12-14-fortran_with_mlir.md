---
layout: post
title:  "FortranをMLIRで表現する話"
date:   2024-12-14 12:00:00 +0900
categories: コンパイラ
tag: llvm mlir
---
この記事は [Fujitsu Advent Calendar 2024](https://qiita.com/advent-calendar/2024/fujitsu) 14日目の記事です。  
なお、本記事は個人の意見に基づくものであり、組織を代表するものではありません。

珍しく人に読まれることを意識して書いていて、柄にもなく敬体です。

# 概要
[LLVM Flang](https://flang.llvm.org/docs/)は内部でのプログラム表現(IR)としてFIRという独自の表現方法を使っています。
これは[MLIR](https://mlir.llvm.org/)の仕組みを利用したものですが、正直なところ、最適化観点では現状MLIRを活用できていないと思っています。
その理由として、ほとんどがFIR独自の表現であり、MLIR付属のdialectはごく一部しか使われないことが挙げられます。

この記事ではFortranプログラムをMLIR付属のdialect**だけ**で表現しようとするとどうなるかを検証します。

# 前提知識
概要を一発で理解できた人はほとんどいないと思うので

## コンパイラとIR
<details markdown="1"><summary>いろいろ書いたら長くなってしまったので折り畳み</summary>

コンパイラが何なのかを知っている人は多いと思います。
`gcc`とか`javac`とかコマンドを打つと動くやつですね。
`python`は違います(インタプリタです)。
コンピュータを動かすには0と1の組み合わせの羅列(機械語)を与える必要がありますが、人間の手でそれを全部やるのはとても厳しいので、ざっくりこんなことをしてほしいという情報(ソースコード)から機械語を生成してくれるプログラムが欲しくなります。
これをやってくれるのがコンパイラ(とアセンブラ)です。

さて、人間が書いたソースコードを機械語に変換するためには、当然ソースコードを解釈できなくてはなりません。
コンパイラは受け取ったソースコードをそのまま文字列として保持しているわけではなく、プログラムを表現できるデータ構造に変換しています。
これを中間表現(Intermediate Representation、略してIR)と呼びます。
このIRには決まった規格などはなく、各コンパイラが扱いやすいよう自由に設計できます。

### LLVM IR

機械語の生成はコンパイラを作ることで楽になりましたが、プログラムの最適化などコンパイラに求められる機能が増えてくると、今度はそのコンパイラを作るのが大変だなとなってきます。
そのため、コンパイラを作るための基盤が欲しくなります。
それがLLVMです。
一般的にコンパイラを3つの構成要素に分解する考え方があり、ソースコードをIRに落とし込むフロントエンド、IRを最適化するミドルエンド、IRをターゲットデバイスの機械語へ変換するバックエンドに分かれます。
このうちLLVMはミドルエンドとバックエンドの機能を提供しています。
そしてその中で扱われるIRがLLVM IRです。

例えば、自作言語のコンパイラが作りたいのなら自作言語をLLVM IRに変換するフロントエンドだけ作ればいいし、自作CPU向けコンパイラが作りたいのならLLVM IRを自作CPUの命令に変換するバックエンドだけ作ればいいことになります。

### MLIR

LLVMによってコンパイラを作るのは楽になりました。
ですが、ターゲットデバイスの種類はそこまで増えるようなものではない一方、自分のプログラムをLLVM IRに変換するニーズはいくらでも増えていきます。
そうなると、今度はフロントエンドを作るのが大変だなとなってきます。
フロントエンドもいくつかの機能に分解できますが、ここではざっくりプログラムを解釈する部分とIRに変換する部分で分けます。
このうち後者をある程度共通化できるのではないかと考えた人たちがいました。
それを担うのがMLIRです。
元々は深層学習コンパイラの開発における車輪の再発明を防ぐ目的で始まったプロジェクトですが、現在ではその利用用途は深層学習だけに留まりません。

MLIRでは独自のIRを作る基盤を提供しています。
この説明だとLLVMの代わりとなるコンパイラ基盤なのかと思われるかもしれませんが、MLIRの出力はLLVM IRであって機械語ではないため、MLIRとLLVMを組み合わせて初めて機械語を生成できます。
MLIRにおける独自のIRは"dialect"と呼ばれ、MLIRのプロジェクトの中だけでもたくさんのdialectが用意されています。
もちろん自分で新たにdialectを作ることもできます。

ちなみに、なぜ直接LLVM IRに変換せずわざわざ別のIRを経由する必要があるのかと言えば、LLVM IRはそこそこ機械語に近いプログラム表現になっているためです。
例えば変数の代入 (e.g. `x = 1`) とかであれば、値をレジスタなりメモリなりにストアすればいいのですが、行列積 (e.g. `numpy.matmul(a, b)`) のアルゴリズムを機械語レベルで書き下せと言われるとまあまあ辛いです。
この例で言いたかったことは、複雑な処理を複雑な処理のままLLVM IRに変換しようとすると大変だということです。
MLIRを使えば複雑な処理をいくつかのパーツに分解し、それぞれ独立してLLVM IRに変換することができるので幾分楽になります。

</details>

## Fortran
Fortranは数値計算が得意なプログラミング言語です。
数値計算ならNumPyで良くない？という声が聞こえてきそうですが、実はNumPyの中ではFortranで書かれたライブラリが使われていたりします。
そういったこともあり、数値計算をごりごり行うシミュレーションの分野でよく使われている言語です。
ちなみにFortran(またはFORTRAN)の歴史は長く、1950年代に登場した最古の高級プログラミング言語なんて言われていますが、時代に合わせて規格改定が継続的に行われており、直近では2023年に最新の規格が制定されている~~イカしたナウい~~言語です。

## LLVM Flang
LLVM Flang(以下Flang)はLLVMをベースとしたFortranコンパイラ(厳密にはFortranフロントエンド)です。
Clangを知っている人もいるかもしれませんが、あれのFortran版です。
現状ではまだ成熟したとは言えない状況ですが、よくあるFortranプログラムであれば問題なく使える程度には開発が進んでいると思います。

このFlangではHLFIR、FIR(、FIRCG)という独自のIRを使ってFortranプログラムをLLVM IRに変換し、LLVMで最適化およびコード生成を行います。
これら独自のIRはMLIRのdialectとして実装されています。
(MLIRのプロジェクトには取り込まれていません。)

### 課題
MLIRの強みはいろんなdialectをごちゃまぜにした状態で部分的にコード変換を適用できるところにあるのですが、現状のFlangは一部を除きすべてFIR dialectのままコード変換されていき、最後にまとめてLLVM IRに変換します。
これによってFortranの言語機能を漏れなく表現できるというメリットもあるのですが、既に存在するMLIRの最適化機能をFlangでは利用できません。
例えばループを表す表現として`scf.for`や`affine.for`といったものがMLIRにあり、これらに対する最適化も存在します。
しかし、Flangでは独自のループ表現(`fir.do_loop`)を用いており、先述の最適化機能はこれに対応していません。
一般的にはTraitやInterfaceの仕組みを使えば、独自の表現であってもうまく対応できるのですが、`fir.do_loop`は独自性が強く、それを難しくしています。
そのため、同様の最適化を適用しようと思うと`fir.do_loop`のためだけの最適化機能が別途必要になる可能性があります。[^duplicate_pass]
これはまさしくMLIRが避けようとしていた車輪の再発明に他なりません。

[^duplicate_pass]: 逆にMLIR側をFIRに歩み寄らせる方法もあるとは思います。受け入れられるかは未知数ですが。

# 検証
お題として用意したFortranソースコードに対してFlangが生成するFIRを書き換え、FIR dialectを根絶した状態で`mlir-opt`と`mlir-translate`を使ってLLVM IRを生成します。
そしてそれをコンパイルし、期待通りの実行結果が得られることを確認します。
Flang独自のランタイムライブラリに関してはそのまま使ってもよいものとします。  
(先ほどあれだけ言っておきながら今回は最適化を何もかけません。)

* 実行コマンド
  * FIR生成
    ```console
    $ flang -fc1 -emit-fir sample.f90 -o sample.fir
    ```
  * LLVM IR生成(例)
    ```console
    $ mlir-opt -pass-pipeline="builtin.module(convert-func-to-llvm)" sample.mlir | mlir-translate --mlir-to-llvmir -o sample.ll
    ```
  * 実行バイナリ生成
    ```console
    $ flang sample.ll -L/path/to/build/lib -lmlir_c_runner_utils
    ```

## お題1: 変数宣言、ループ
以下のFortranプログラムからLLVM IRを生成します。

```fortran
implicit none
integer :: i, sum = 0
do i=1,100
  sum = sum + i
end do
print *, sum
end
```

<details markdown="1"><summary>(参考)Flangが生成するFIR</summary>

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() {
    %c6_i32 = arith.constant 6 : i32
    %c1 = arith.constant 1 : index
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %1 = fir.declare %0 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %2 = fir.address_of(@_QFEsum) : !fir.ref<i32>
    %3 = fir.declare %2 {uniq_name = "_QFEsum"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %4 = fir.convert %c1_i32 : (i32) -> index
    %5 = fir.convert %c100_i32 : (i32) -> index
    %6 = fir.convert %4 : (index) -> i32
    %7:2 = fir.do_loop %arg0 = %4 to %5 step %c1 iter_args(%arg1 = %6) -> (index, i32) {
      fir.store %arg1 to %1 : !fir.ref<i32>
      %14 = fir.load %3 : !fir.ref<i32>
      %15 = fir.load %1 : !fir.ref<i32>
      %16 = arith.addi %14, %15 : i32
      fir.store %16 to %3 : !fir.ref<i32>
      %17 = arith.addi %arg0, %c1 : index
      %18 = fir.convert %c1 : (index) -> i32
      %19 = fir.load %1 : !fir.ref<i32>
      %20 = arith.addi %19, %18 : i32
      fir.result %17, %20 : index, i32
    }
    fir.store %7#1 to %1 : !fir.ref<i32>
    %8 = fir.address_of(@_QQclXc01d1ee3012d47935c98f9e1e8a5598c) : !fir.ref<!fir.char<1,50>>
    %9 = fir.convert %8 : (!fir.ref<!fir.char<1,50>>) -> !fir.ref<i8>
    %10 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %9, %c6_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.call @_FortranAioOutputInteger32(%10, %11) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
    %13 = fir.call @_FortranAioEndIoStatement(%10) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  fir.global internal @_QFEsum : i32 {
    %c0_i32 = arith.constant 0 : i32
    fir.has_value %c0_i32 : i32
  }
  func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclXc01d1ee3012d47935c98f9e1e8a5598c constant : !fir.char<1,50> {
    %0 = fir.string_lit "/home/yus3710/work/llvm/flang/fir2mlir/level1.f90\00"(50) : !fir.char<1,50>
    fir.has_value %0 : !fir.char<1,50>
  }
  func.func private @_FortranAioOutputInteger32(!fir.ref<i8>, i32) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
    fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) fastmath<contract> : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) -> ()
    fir.call @_QQmain() fastmath<contract> : () -> ()
    fir.call @_FortranAProgramEndStatement() fastmath<contract> : () -> ()
    return %c0_i32 : i32
  }
}
```

</details>

これに対して用意したMLIRは以下です。

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() {
    %c6_i32 = arith.constant 6 : i32
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c100_i32 = arith.constant 100 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = memref.alloca() {bindc_name = "i", uniq_name = "_QFEi"} : memref<i32>
    //%1 = fir.declare %0 {uniq_name = "_QFEi"} : (memref<i32>) -> memref<i32>
    %2 = memref.get_global @_QFEsum : memref<i32>
    //%3 = fir.declare %2 {uniq_name = "_QFEsum"} : (memref<i32>) -> memref<i32>
    %4 = arith.index_cast %c1_i32 : i32 to index
    %5 = arith.index_cast %c100_i32 : i32 to index
    %6 = arith.index_cast %4 : index to i32
    %7 = scf.for %arg0 = %c0 to %5 step %c1 iter_args(%arg1 = %6) -> (i32) {
      memref.store %arg1, %0[] : memref<i32>
      %12 = memref.load %2[] : memref<i32>
      %13 = memref.load %0[] : memref<i32>
      %14 = arith.addi %12, %13 : i32
      memref.store %14, %2[] : memref<i32>
      %15 = arith.index_cast %c1 : index to i32
      %16 = memref.load %0[] : memref<i32>
      %17 = arith.addi %16, %15 : i32
      scf.yield %17 : i32
    }
    memref.store %7, %0[] : memref<i32>
    //%8 = memref.get_global @fmt : memref<3xi8>
    %8 = llvm.mlir.addressof @fmt : !llvm.ptr
    %c0_i64 = arith.constant 0 : i64
    %9 = llvm.getelementptr %8[%c0_i64, %c0_i64] : (!llvm.ptr, i64, i64) -> !llvm.ptr, !llvm.array<4 x i8>
    %10 = memref.load %2[] : memref<i32>
    %11 = llvm.call @printf(%9, %10) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr, i32) -> i32
    return
  }
  memref.global "private" @_QFEsum : memref<i32> = dense<0>
  //memref.global "private" constant @fmt : memref<3xi8> = dense<"%d\0A\00">
  llvm.mlir.global internal constant @fmt("%d\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.zero : !llvm.ptr
    func.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) {fastmathFlags = #llvm.fastmath<contract>} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
    func.call @_QQmain() {fastmathFlags = #llvm.fastmath<contract>} : () -> ()
    func.call @_FortranAProgramEndStatement() {fastmathFlags = #llvm.fastmath<contract>} : () -> ()
    return %c0_i32 : i32
  }
}

```

これを以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),finalize-memref-to-llvm,canonicalize,convert-func-to-llvm,reconcile-unrealized-casts)" level1.mlir | mlir-translate --mlir-to-llvmir -o level1.ll
```

### 解説
* `fir.alloca`は`memref.alloca`に変換します
* `fir.convert`は対応するキャスト(今回は`arith.index_cast`)に変換します
* `fir.do_loop`は`scf.for`に変換します
  * Fortranの言語規格を遵守するにはループの回転数を最初に計算し、その分だけ回転する必要があります。今回は100回転することが明らかなので[0, 100)の範囲で回転するようにループのパラメータを設定します。
    * 以下のFortranとCのプログラムは、等価ではありません。Fortranの方は、言語規格に則れば(127-(-127))/1+1=255[^overflow]回転して終了するはずです。[^gfort_loop]Cの方は`i`が128以上にはならないため無限ループになります。
      ```fortran
      do i=-127_1,127_1
        ! 何か適当な処理
      end do
      ```
      ```c
      for (char i = -127; i <= 127; i++) {
        // 何か適当な処理
      }
      ```
  * ループ変数は`iter_args`の方で更新していきます。
* グローバル変数は`fir.global`の代わりに`memref.global`で定義し、`fir.address_of`の代わりに`memref.get_global`でアドレスを取得するようにします
* `fir.zero_bits`はおそらくNULLポインタを作るための操作なのですが、`llvm.mlir.zero`しか使えそうなものがなかったのでこれに置き換えます
  * MLIRにはポインタ型が存在しません
* PRINT文に対応する関数は、今回は`printf`に変換します
* `fir.declare`に関しては完全にFlang独自の概念(変数の宣言文)なので一旦無視します

### 考察
* `fir.global`を`memref.global`へ変換する際に属性を`internal`から`private`へ変換していますが、`llvm.mlir.global`はinternalもprivateもあるようなので、`fir.global`を使った方がいいのかもしれません。
* MLIRに`string`型なる文字列型を導入してもよさそうと思いました。
  * むしろなぜないのでしょうか…？
  * これが理由なのか、文字列リテラルは`memref.global`でうまくいかなかったため`llvm.mlir.global`で定義しています。

[^overflow]: 8bitで計算するとオーバーフローするのですが、言語規格には別に8bitのまま計算しろとは書かれていないので、Flangではなるべく正確に計算するために1wordで計算します。
[^gfort_loop]: といいつつGFortranはCみたいなコードを生成してるようですね…

## お題2: 配列(超単純ver.)
以下のFortranプログラムからLLVM IRを生成します。

```fortran
implicit none
integer :: i
integer, parameter :: n = 10
real :: a(n), b(n)
data b/n*1/

do i=1,n,2
  a(i) = b(i) * i
end do
print *, a
end
```

<details markdown="1"><summary>(参考)Flangが生成するFIR</summary>

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() {
    %c11_i32 = arith.constant 11 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c10 = arith.constant 10 : index
    %0 = fir.address_of(@_QFEa) : !fir.ref<!fir.array<10xf32>>
    %1 = fir.shape %c10 : (index) -> !fir.shape<1>
    %2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %3 = fir.address_of(@_QFEb) : !fir.ref<!fir.array<10xf32>>
    %4 = fir.shape %c10 : (index) -> !fir.shape<1>
    %5 = fir.declare %3(%4) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %6 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %7 = fir.declare %6 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = fir.address_of(@_QFECn) : !fir.ref<i32>
    %9 = fir.declare %8 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = fir.convert %c1_i32 : (i32) -> index
    %11 = fir.convert %c10_i32 : (i32) -> index
    %12 = fir.convert %c2_i32 : (i32) -> index
    %13 = fir.convert %10 : (index) -> i32
    %14:2 = fir.do_loop %arg0 = %10 to %11 step %12 iter_args(%arg1 = %13) -> (index, i32) {
      fir.store %arg1 to %7 : !fir.ref<i32>
      %23 = fir.load %7 : !fir.ref<i32>
      %24 = fir.convert %23 : (i32) -> i64
      %25 = fir.array_coor %5(%4) %24 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      %26 = fir.load %25 : !fir.ref<f32>
      %27 = fir.load %7 : !fir.ref<i32>
      %28 = fir.convert %27 : (i32) -> f32
      %29 = arith.mulf %26, %28 fastmath<contract> : f32
      %30 = fir.load %7 : !fir.ref<i32>
      %31 = fir.convert %30 : (i32) -> i64
      %32 = fir.array_coor %2(%1) %31 : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>, i64) -> !fir.ref<f32>
      fir.store %29 to %32 : !fir.ref<f32>
      %33 = arith.addi %arg0, %12 : index
      %34 = fir.convert %12 : (index) -> i32
      %35 = fir.load %7 : !fir.ref<i32>
      %36 = arith.addi %35, %34 : i32
      fir.result %33, %36 : index, i32
    }
    fir.store %14#1 to %7 : !fir.ref<i32>
    %15 = fir.address_of(@_QQclX33226f0b9a52a8f32dd42d6ea05e9142) : !fir.ref<!fir.char<1,50>>
    %16 = fir.convert %15 : (!fir.ref<!fir.char<1,50>>) -> !fir.ref<i8>
    %17 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %16, %c11_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %18 = fir.shape %c10 : (index) -> !fir.shape<1>
    %19 = fir.embox %2(%18) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
    %20 = fir.convert %19 : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
    %21 = fir.call @_FortranAioOutputDescriptor(%17, %20) fastmath<contract> : (!fir.ref<i8>, !fir.box<none>) -> i1
    %22 = fir.call @_FortranAioEndIoStatement(%17) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  fir.global internal @_QFEa : !fir.array<10xf32> {
    %0 = fir.zero_bits !fir.array<10xf32>
    fir.has_value %0 : !fir.array<10xf32>
  }
  fir.global internal @_QFEb(dense<1.000000e+00> : tensor<10xf32>) : !fir.array<10xf32>
  fir.global internal @_QFECn constant : i32 {
    %c10_i32 = arith.constant 10 : i32
    fir.has_value %c10_i32 : i32
  }
  func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
  fir.global linkonce @_QQclX33226f0b9a52a8f32dd42d6ea05e9142 constant : !fir.char<1,50> {
    %0 = fir.string_lit "/home/yus3710/work/llvm/flang/fir2mlir/level2.f90\00"(50) : !fir.char<1,50>
    fir.has_value %0 : !fir.char<1,50>
  }
  func.func private @_FortranAioOutputDescriptor(!fir.ref<i8>, !fir.box<none>) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
    fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) fastmath<contract> : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) -> ()
    fir.call @_QQmain() fastmath<contract> : () -> ()
    fir.call @_FortranAProgramEndStatement() fastmath<contract> : () -> ()
    return %c0_i32 : i32
  }
}
```

</details>

これに対して用意したMLIRは以下です。

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c-1 = arith.constant -1 : index
    %c0 = arith.constant 0 : index
    %c11_i32 = arith.constant 11 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %0 = memref.get_global @_QFEa : memref<10xf32>
    //%1 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %box = llvm.alloca %c1_i32 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = memref.get_global @_QFEb : memref<10xf32>
    //%4 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%5 = fir.declare %3(%4) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %6 = memref.alloca() {bindc_name = "i", uniq_name = "_QFEi"} : memref<i32>
    //%7 = fir.declare %6 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = memref.get_global @_QFECn : memref<i32>
    //%9 = fir.declare %8 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = arith.index_cast %c1_i32 : i32 to index
    %11 = arith.index_cast %c10_i32 : i32 to index
    %12 = arith.index_cast %c2_i32 : i32 to index
    %13 = arith.index_cast %10 : index to i32
    %14 = scf.for %arg3 = %c0 to %c5 step %c1 iter_args(%arg4 = %13) -> (i32) {
      memref.store %arg4, %6[] : memref<i32>
      %23 = memref.load %6[] : memref<i32>
      %24 = arith.index_cast %23 : i32 to index
      %b_mem = memref.subview %3[%c-1][%c10][1] : memref<10xf32> to memref<?xf32, strided<[1], offset: ?>>
      %25 = memref.cast %b_mem : memref<?xf32, strided<[1], offset: ?>> to memref<10xf32, strided<[1], offset: 1>>
      %26 = memref.load %25[%24] : memref<10xf32, strided<[1], offset: 1>>
      %27 = memref.load %6[] : memref<i32>
      %28 = arith.sitofp %27 : i32 to f32
      %29 = arith.mulf %26, %28 fastmath<contract> : f32
      %30 = memref.load %6[] : memref<i32>
      %31 = arith.index_cast %30 : i32 to index
      %32 = memref.subview %0[%c-1][%c10][%c1] : memref<10xf32> to memref<?xf32, strided<[?], offset: ?>>
      //%32 = memref.cast %a_mem : memref<?xf32, strided<[?], offset: ?>> to memref<10xf32, strided<[1], offset: 1>>
      memref.store %29, %32[%31] : memref<?xf32, strided<[?], offset: ?>>
      %34 = arith.index_cast %12 : index to i32
      %35 = memref.load %6[] : memref<i32>
      %36 = arith.addi %35, %34 : i32
      scf.yield %36 : i32
    }
    memref.store %14, %6[] : memref<i32>
    %16 = llvm.mlir.addressof @_QQclX33226f0b9a52a8f32dd42d6ea05e9142 : !llvm.ptr
    %17 = func.call @_FortranAioBeginExternalListOutput(%c6_i32, %16, %c11_i32) {fastmathFlags = #llvm.fastmath<contract>} : (i32, !llvm.ptr, i32) -> !llvm.ptr
    //%18 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%19 = fir.embox %2(%18) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
    %desc0 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %null = llvm.mlir.zero : !llvm.ptr
    %f32 = llvm.getelementptr %null[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %elem_len = llvm.ptrtoint %f32 : !llvm.ptr to i64
    %desc1 = llvm.insertvalue %elem_len, %desc0[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %version = llvm.mlir.constant(20180515 : i32) : i32
    %desc2 = llvm.insertvalue %version, %desc1[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %rank = llvm.mlir.constant(1 : i32) : i32
    %rank_i8 = llvm.trunc %rank : i32 to i8
    %desc3 = llvm.insertvalue %rank_i8, %desc2[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %type = llvm.mlir.constant(27 : i32) : i32
    %type_i8 = llvm.trunc %type : i32 to i8
    %desc4 = llvm.insertvalue %type_i8, %desc3[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %attr = llvm.mlir.constant(0 : i32) : i32
    %attr_i8 = llvm.trunc %attr : i32 to i8
    %desc5 = llvm.insertvalue %attr_i8, %desc4[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %extra = llvm.mlir.constant(0 : i32) : i32
    %extra_i8 = llvm.trunc %extra : i32 to i8
    %desc = llvm.insertvalue %extra_i8, %desc5[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc7 = llvm.insertvalue %base_addr, %desc6[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc8 = llvm.insertvalue %lower_bound, %desc7[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc9 = llvm.insertvalue %extent, %desc8[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc = llvm.insertvalue %sm, %desc9[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%20 = fir.convert %19 : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
    llvm.store %desc, %box : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %21 = func.call @_FortranAioOutputDescriptor(%17, %box) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr, !llvm.ptr) -> i1
    %22 = func.call @_FortranAioEndIoStatement(%17) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr) -> i32
    return %c0_i32 : i32
  }
  memref.global "private" @_QFEa : memref<10xf32> = dense<0.000000e+00>
  memref.global "private" @_QFEb : memref<10xf32> = dense<1.000000e+00>
  memref.global "private" constant @_QFECn : memref<i32> = dense<10>
  func.func private @_FortranAioBeginExternalListOutput(i32, !llvm.ptr, i32) -> !llvm.ptr attributes {fir.io, fir.runtime}
  llvm.mlir.global linkonce constant @_QQclX33226f0b9a52a8f32dd42d6ea05e9142() comdat(@__llvm_comdat::@_QQclX33226f0b9a52a8f32dd42d6ea05e9142) {addr_space = 0 : i32} : !llvm.array<50 x i8> {
    %0 = llvm.mlir.constant("/home/yus3710/work/llvm/flang/fir2mlir/level2.f90\00") : !llvm.array<50 x i8>
    llvm.return %0 : !llvm.array<50 x i8>
  }
  llvm.comdat @__llvm_comdat {
    llvm.comdat_selector @_QQclX33226f0b9a52a8f32dd42d6ea05e9142 any
  }
  func.func private @_FortranAioOutputDescriptor(!llvm.ptr, !llvm.ptr) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!llvm.ptr) -> i32 attributes {fir.io, fir.runtime}
}
```

これを以下のコマンドでLLVM IRに変換します。

```console
$ mlir-opt -pass-pipeline="builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),expand-strided-metadata,finalize-memref-to-llvm,canonicalize,cse,convert-func-to-llvm,reconcile-unrealized-casts)" level1.mlir | mlir-translate --mlir-to-llvmir -o level1.ll
```

### 解説
* 配列は`memref`で表現します
  * `tensor`かと思いましたが、`tensor`はグローバル変数になれません。
* `fir.shape`は今回不要だったので無視します
  * 今後shape dialectの表現で代用しないといけない場面が出てくるかもしれません。
* `fir.array_coor`を`memref.subview`に変換します
  * Fortranでは添字の範囲を自由に決められます。そのため、添字の下限値を指定したときに先頭を指すように帳尻合わせをする必要があります。それを表現できるのが`memref.subview`です。
    * ただし、`memref.cast`を使っておかないと情報がごっそり抜け落ちて最適化を適用するうえで不利になると思います。(今回はそこまで検証できていませんが)
    * 本来、`memref.subview`は部分配列の表現に使われるものだと思います。が、配列自身も部分配列と言えなくもないので使っています。
* PRINT文に対応する関数は、今回はFlang独自のランタイムルーチンをそのまま流用します
  * memrefをそのまま渡せないためランタイム側が受け取れる形に変換する必要があります。
    * Cだとポインタと配列は表裏一体ですが、ほかの言語ではそうではないこともあります。その場合、先頭アドレスだけでなく、配列の形状などの情報も一緒に渡す場合があります。この情報の集合体をDescriptorと呼んでいます。
    * このフォーマットが[MLIR](https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types)の`memref`と[Flang](https://flang.llvm.org/docs/RuntimeDescriptor.html#proposal)の`fir.box`で違います。
    * この変換は`finalize-memref-to-llvm`を通した後に手で実施しています。

### 考察
* Descriptorに関しては、memrefの方に完全移行するのはなんとなく難しいのではないかと思います。
  * 例えばポインタかALLOCATABLEかそのどちらでもないか、という情報はmemrefの方で持つ意味がなさそうです。
  * なので、MLIRのDescriptorからFlangのDescriptorに変換するのを`fir.embox`で出来るようになるといいのかなと思いました。
  * 似たような話として、昔は複素数型がMLIRにもFIRにもあって、両方を受け入れて処理できるようにしていたようです。(後にMLIRの方に統一されましたが)
* `fir.global`は先述の件と合わせると、大人しくそのまま使うのが良いかもしれません。(cf. ml_program dialect)
* 以下のように`affine_map`を使っても変換できます
  * この方法でLLVM IRに変換すると、アドレス計算の命令列が元々Flangの生成しているLLVM IRに近くなります。
  * ただし先述の言語規格を無視してループ変数をそのままパラメータとして渡すことになるため、affine dialectの最適化を一通り適用したらどこかのタイミングで言語規格に合わせて直す必要がありそうです。
    * AffineLoopNormalizeなんてものもあるみたいですが…

<details markdown="1"><summary>(参考)affine dialectを使った例</summary>

```
#array_access = affine_map<(d0)[s0, s1] -> (d0 * s1 - s0)>
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c5 = arith.constant 5 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c11_i32 = arith.constant 11 : i32
    %c6_i32 = arith.constant 6 : i32
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10 = arith.constant 10 : index
    %0 = memref.get_global @_QFEa : memref<10xf32>
    //%1 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%2 = fir.declare %0(%1) {uniq_name = "_QFEa"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %box = llvm.alloca %c1_i32 x !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)> {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = memref.get_global @_QFEb : memref<10xf32>
    //%4 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%5 = fir.declare %3(%4) {uniq_name = "_QFEb"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<10xf32>>
    %6 = memref.alloca() {bindc_name = "i", uniq_name = "_QFEi"} : memref<i32>
    //%7 = fir.declare %6 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %8 = memref.get_global @_QFECn : memref<i32>
    //%9 = fir.declare %8 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QFECn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = arith.index_cast %c1_i32 : i32 to index
    %11 = arith.index_cast %c10_i32 : i32 to index
    %12 = arith.index_cast %c2_i32 : i32 to index
    %13 = arith.index_cast %c11_i32 : i32 to index
    // The following violates the Fortran standard.
    %14 = affine.for %arg3 = %10 to %13 step 2 iter_args(%arg4 = %c5) -> (index) {
      %23 = arith.index_cast %arg3 : index to i32
      memref.store %23, %6[] : memref<i32>
      //%25 = affine.apply #array_access (%arg3)[%c1, %c1]
      //%26 = memref.load %3[%25] : memref<10xf32>
      %26 = affine.load %3[%arg3 - symbol(%c1)] : memref<10xf32>
      %28 = arith.sitofp %23 : i32 to f32
      %29 = arith.mulf %26, %28 fastmath<contract> : f32
      //%32 = affine.apply #array_access (%arg3)[%c1, %c1]
      //memref.store %29, %0[%32] : memref<10xf32>
      affine.store %29, %0[%arg3 - symbol(%c1)] : memref<10xf32>
      //%34 = arith.index_cast %12 : index to i32
      //%35 = memref.load %6[] : memref<i32>
      //%36 = arith.addi %35, %34 : i32
      %36 = arith.subi %arg4, %c1 : index
      affine.yield %36 : index
    }
    //memref.store %14, %6[] : memref<i32>
    %16 = llvm.mlir.addressof @_QQclX33226f0b9a52a8f32dd42d6ea05e9142 : !llvm.ptr
    %17 = func.call @_FortranAioBeginExternalListOutput(%c6_i32, %16, %c11_i32) {fastmathFlags = #llvm.fastmath<contract>} : (i32, !llvm.ptr, i32) -> !llvm.ptr
    //%18 = fir.shape %c10 : (index) -> !fir.shape<1>
    //%19 = fir.embox %2(%18) : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> !fir.box<!fir.array<10xf32>>
    %desc0 = llvm.mlir.undef : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %null = llvm.mlir.zero : !llvm.ptr
    %f32 = llvm.getelementptr %null[1] : (!llvm.ptr) -> !llvm.ptr, f32
    %elem_len = llvm.ptrtoint %f32 : !llvm.ptr to i64
    %desc1 = llvm.insertvalue %elem_len, %desc0[1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %version = llvm.mlir.constant(20180515 : i32) : i32
    %desc2 = llvm.insertvalue %version, %desc1[2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %rank = llvm.mlir.constant(1 : i32) : i32
    %rank_i8 = llvm.trunc %rank : i32 to i8
    %desc3 = llvm.insertvalue %rank_i8, %desc2[3] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %type = llvm.mlir.constant(27 : i32) : i32
    %type_i8 = llvm.trunc %type : i32 to i8
    %desc4 = llvm.insertvalue %type_i8, %desc3[4] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %attr = llvm.mlir.constant(0 : i32) : i32
    %attr_i8 = llvm.trunc %attr : i32 to i8
    %desc5 = llvm.insertvalue %attr_i8, %desc4[5] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    %extra = llvm.mlir.constant(0 : i32) : i32
    %extra_i8 = llvm.trunc %extra : i32 to i8
    %desc = llvm.insertvalue %extra_i8, %desc5[6] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc7 = llvm.insertvalue %base_addr, %desc6[0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc8 = llvm.insertvalue %lower_bound, %desc7[7, 0, 0] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc9 = llvm.insertvalue %extent, %desc8[7, 0, 1] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%desc = llvm.insertvalue %sm, %desc9[7, 0, 2] : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>
    //%20 = fir.convert %19 : (!fir.box<!fir.array<10xf32>>) -> !fir.box<none>
    llvm.store %desc, %box : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8, array<1 x array<3 x i64>>)>, !llvm.ptr
    %21 = func.call @_FortranAioOutputDescriptor(%17, %box) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr, !llvm.ptr) -> i1
    %22 = func.call @_FortranAioEndIoStatement(%17) {fastmathFlags = #llvm.fastmath<contract>} : (!llvm.ptr) -> i32
    return %c0_i32 : i32
  }
  memref.global "private" @_QFEa : memref<10xf32> = dense<0.000000e+00>
  memref.global "private" @_QFEb : memref<10xf32> = dense<1.000000e+00>
  memref.global "private" constant @_QFECn : memref<i32> = dense<10>
  func.func private @_FortranAioBeginExternalListOutput(i32, !llvm.ptr, i32) -> !llvm.ptr attributes {fir.io, fir.runtime}
  llvm.mlir.global linkonce constant @_QQclX33226f0b9a52a8f32dd42d6ea05e9142() comdat(@__llvm_comdat::@_QQclX33226f0b9a52a8f32dd42d6ea05e9142) {addr_space = 0 : i32} : !llvm.array<50 x i8> {
    %0 = llvm.mlir.constant("/home/yus3710/work/llvm/flang/fir2mlir/level2.f90\00") : !llvm.array<50 x i8>
    llvm.return %0 : !llvm.array<50 x i8>
  }
  llvm.comdat @__llvm_comdat {
    llvm.comdat_selector @_QQclX33226f0b9a52a8f32dd42d6ea05e9142 any
  }
  func.func private @_FortranAioOutputDescriptor(!llvm.ptr, !llvm.ptr) -> i1 attributes {fir.io, fir.runtime}
  func.func private @_FortranAioEndIoStatement(!llvm.ptr) -> i32 attributes {fir.io, fir.runtime}
}
```

</details>

## お題3: 派生型
以下のFortranプログラムからLLVM IRを生成します。

```fortran
implicit none
integer :: sum = 0
type score
  integer :: lang
  integer :: math
  integer :: eng
end type
type(score) :: taro = score(80,70,90)

sum = sum + taro%lang
sum = sum + taro%math
sum = sum + taro%eng

print *, sum
end
```

<details markdown="1"><summary>(参考)Flangが生成するFIR</summary>

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() {
    %c14_i32 = arith.constant 14 : i32
    %c6_i32 = arith.constant 6 : i32
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %0 = fir.address_of(@_QFE.n.lang) : !fir.ref<!fir.char<1,4>>
    %1 = fir.declare %0 typeparams %c4 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.lang"} : (!fir.ref<!fir.char<1,4>>, index) -> !fir.ref<!fir.char<1,4>>
    %2 = fir.address_of(@_QFE.n.math) : !fir.ref<!fir.char<1,4>>
    %3 = fir.declare %2 typeparams %c4 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.math"} : (!fir.ref<!fir.char<1,4>>, index) -> !fir.ref<!fir.char<1,4>>
    %4 = fir.address_of(@_QFE.n.eng) : !fir.ref<!fir.char<1,3>>
    %5 = fir.declare %4 typeparams %c3 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.eng"} : (!fir.ref<!fir.char<1,3>>, index) -> !fir.ref<!fir.char<1,3>>
    %6 = fir.address_of(@_QFE.n.score) : !fir.ref<!fir.char<1,5>>
    %7 = fir.declare %6 typeparams %c5 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.n.score"} : (!fir.ref<!fir.char<1,5>>, index) -> !fir.ref<!fir.char<1,5>>
    %8 = fir.address_of(@_QFEsum) : !fir.ref<i32>
    %9 = fir.declare %8 {uniq_name = "_QFEsum"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %10 = fir.address_of(@_QFEtaro) : !fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>
    %11 = fir.declare %10 {uniq_name = "_QFEtaro"} : (!fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>) -> !fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>
    %12 = fir.address_of(@_QFE.c.score) : !fir.ref<!fir.array<3x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>
    %13 = fir.shape_shift %c0, %c3 : (index, index) -> !fir.shapeshift<1>
    %14 = fir.declare %12(%13) {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.c.score"} : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<3x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>
    %15 = fir.embox %14(%13) : (!fir.ref<!fir.array<3x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<3x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>
    %16 = fir.address_of(@_QFE.dt.score) : !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>
    %17 = fir.declare %16 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFE.dt.score"} : (!fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>) -> !fir.ref<!fir.type<_QM__fortran_type_infoTderivedtype{binding:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTbinding{proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>}>>>>,name:!fir.box<!fir.ptr<!fir.char<1,?>>>,sizeinbytes:i64,uninstantiated:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,kindparameter:!fir.box<!fir.ptr<!fir.array<?xi64>>>,lenparameterkind:!fir.box<!fir.ptr<!fir.array<?xi8>>>,component:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,genre:i8,category:i8,kind:i8,rank:i8,__padding0:!fir.array<4xi8>,offset:i64,characterlen:!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>,derived:!fir.box<!fir.ptr<!fir.type<_QM__fortran_type_infoTderivedtype>>>,lenvalue:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,bounds:!fir.box<!fir.ptr<!fir.array<?x?x!fir.type<_QM__fortran_type_infoTvalue{genre:i8,__padding0:!fir.array<7xi8>,value:i64}>>>>,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>}>>>>,procptr:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTprocptrcomponent{name:!fir.box<!fir.ptr<!fir.char<1,?>>>,offset:i64,initialization:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,special:!fir.box<!fir.ptr<!fir.array<?x!fir.type<_QM__fortran_type_infoTspecialbinding{which:i8,isargdescriptorset:i8,istypebound:i8,isargcontiguousset:i8,__padding0:!fir.array<4xi8>,proc:!fir.type<_QM__fortran_builtinsT__builtin_c_funptr{__address:i64}>}>>>>,specialbitset:i32,hasparent:i8,noinitializationneeded:i8,nodestructionneeded:i8,nofinalizationneeded:i8,__padding0:!fir.array<4xi8>}>>
    %18 = fir.load %9 : !fir.ref<i32>
    %19 = fir.field_index lang, !fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>
    %20 = fir.coordinate_of %11, %19 : (!fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>, !fir.field) -> !fir.ref<i32>
    %21 = fir.load %20 : !fir.ref<i32>
    %22 = arith.addi %18, %21 : i32
    fir.store %22 to %9 : !fir.ref<i32>
    %23 = fir.load %9 : !fir.ref<i32>
    %24 = fir.field_index math, !fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>
    %25 = fir.coordinate_of %11, %24 : (!fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>, !fir.field) -> !fir.ref<i32>
    %26 = fir.load %25 : !fir.ref<i32>
    %27 = arith.addi %23, %26 : i32
    fir.store %27 to %9 : !fir.ref<i32>
    %28 = fir.load %9 : !fir.ref<i32>
    %29 = fir.field_index eng, !fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>
    %30 = fir.coordinate_of %11, %29 : (!fir.ref<!fir.type<_QFTscore{lang:i32,math:i32,eng:i32}>>, !fir.field) -> !fir.ref<i32>
    %31 = fir.load %30 : !fir.ref<i32>
    %32 = arith.addi %28, %31 : i32
    fir.store %32 to %9 : !fir.ref<i32>
    %33 = fir.address_of(@_QQclX7ae47400e92558abd198dc2f940fb3e1) : !fir.ref<!fir.char<1,50>>
    %34 = fir.convert %33 : (!fir.ref<!fir.char<1,50>>) -> !fir.ref<i8>
    %35 = fir.call @_FortranAioBeginExternalListOutput(%c6_i32, %34, %c14_i32) fastmath<contract> : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
    %36 = fir.load %9 : !fir.ref<i32>
    %37 = fir.call @_FortranAioOutputInteger32(%35, %36) fastmath<contract> : (!fir.ref<i8>, i32) -> i1
    %38 = fir.call @_FortranAioEndIoStatement(%35) fastmath<contract> : (!fir.ref<i8>) -> i32
    return
  }
  ...
}
```

</details>

これに対して用意したMLIRはありません。

### 解説
派生型を表現できそうなのは`tuple`だと思うのですが、`tuple`には関連する操作が定義されていないほか、変換規則も存在しません。
これは意図的にそうしているそうです。
`tuple`にはどんなdialectの型を組み合わせても問題ないはずで、それをMLIR側で全部考慮するのは不可能だからです。
言い換えると、必要なものがあるなら`tuple`を使う側の責任で実装しなさいということです。

### 考察
さすがに`llvm.struct`を使えば表現できるとは思いますが、`fir.type`の方がある程度抽象化されていて表現しやすいと思います。
例えば派生型の成分は連想配列の感覚で取得できます。

## お題4: 形状引継ぎ配列、部分配列
以下のFortranプログラムからLLVM IRを生成します。

```fortran
! 主プログラムは別ファイルにあります
! そちらには手を入れません
subroutine sub(a,b)
  implicit none
  real :: a(1:,2:), b(3:,4:)

  a = b(:size(a,1)+lbound(b,1)-1,:size(a,2)+lbound(b,2)-1)
end subroutine
```

<details markdown="1"><summary>(参考)Flangが生成するFIR</summary>

```
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QPsub(%arg0: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %arg1: !fir.box<!fir.array<?x?xf32>> {fir.bindc_name = "b"}) {
    %0 = fir.alloca !fir.box<!fir.array<?x?xf32>>
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    %1 = fir.dummy_scope : !fir.dscope
    %2 = fir.convert %c1_i64 : (i64) -> index
    %3 = fir.convert %c2_i64 : (i64) -> index
    %4 = fir.shift %2, %3 : (index, index) -> !fir.shift<2>
    %5 = fir.declare %arg0(%4) dummy_scope %1 {uniq_name = "_QFsubEa"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
    %6 = fir.rebox %5(%4) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
    %7 = fir.convert %c3_i64 : (i64) -> index
    %8 = fir.convert %c4_i64 : (i64) -> index
    %9 = fir.shift %7, %8 : (index, index) -> !fir.shift<2>
    %10 = fir.declare %arg1(%9) dummy_scope %1 {uniq_name = "_QFsubEb"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
    %11 = fir.rebox %10(%9) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
    %12:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %13 = fir.convert %12#1 : (index) -> i64
    %14 = fir.convert %13 : (i64) -> i32
    %15:3 = fir.box_dims %11, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %16 = arith.cmpi eq, %15#1, %c0 : index
    %17 = fir.convert %c1_i32 : (i32) -> index
    %18 = arith.select %16, %17, %7 : index
    %19 = fir.convert %18 : (index) -> i32
    %20 = arith.addi %14, %19 : i32
    %21 = arith.subi %20, %c1_i32 : i32
    %22 = fir.convert %21 : (i32) -> i64
    %23 = fir.convert %22 : (i64) -> index
    %24 = arith.subi %23, %7 : index
    %25 = arith.addi %24, %c1 : index
    %26 = arith.cmpi sgt, %25, %c0 : index
    %27 = arith.select %26, %25, %c0 : index
    %28:3 = fir.box_dims %6, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %29 = fir.convert %28#1 : (index) -> i64
    %30 = fir.convert %29 : (i64) -> i32
    %31:3 = fir.box_dims %11, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %32 = arith.cmpi eq, %31#1, %c0 : index
    %33 = fir.convert %c1_i32 : (i32) -> index
    %34 = arith.select %32, %33, %8 : index
    %35 = fir.convert %34 : (index) -> i32
    %36 = arith.addi %30, %35 : i32
    %37 = arith.subi %36, %c1_i32 : i32
    %38 = fir.convert %37 : (i32) -> i64
    %39 = fir.convert %38 : (i64) -> index
    %40 = arith.subi %39, %8 : index
    %41 = arith.addi %40, %c1 : index
    %42 = arith.cmpi sgt, %41, %c0 : index
    %43 = arith.select %42, %41, %c0 : index
    %44 = fir.shape %27, %43 : (index, index) -> !fir.shape<2>
    %45 = fir.undefined index
    %46 = fir.slice %7, %23, %c1, %8, %39, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
    %47 = fir.rebox %11(%9) [%46] : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
    fir.store %6 to %0 : !fir.ref<!fir.box<!fir.array<?x?xf32>>>
    %48:3 = fir.box_dims %6, %c0 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    %49:3 = fir.box_dims %6, %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
    fir.do_loop %arg2 = %c1 to %48#1 step %c1 unordered {
      fir.do_loop %arg3 = %c1 to %49#1 step %c1 unordered {
        %50 = fir.array_coor %47 %arg3, %arg2 : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
        %51 = fir.load %50 : !fir.ref<f32>
        %52 = arith.addi %arg2, %c1 : index
        %53 = fir.array_coor %6(%4) %arg3, %52 : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, index, index) -> !fir.ref<f32>
        fir.store %51 to %53 : !fir.ref<f32>
      }
    }
    return
  }
}
```

</details>

すみません、MLIRの用意は間に合いませんでした…  
一応、格闘の痕跡は残しておきます。

```
// mlir-opt -pass-pipeline="builtin.module(one-shot-bufferize{bufferize-function-boundaries},buffer-deallocation-pipeline,func.func(convert-scf-to-cf,convert-arith-to-llvm),expand-strided-metadata,finalize-memref-to-llvm,canonicalize)"
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<f80, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi64>>, #dlti.dl_entry<i1, dense<8> : vector<2xi64>>, #dlti.dl_entry<i8, dense<8> : vector<2xi64>>, #dlti.dl_entry<i16, dense<16> : vector<2xi64>>, #dlti.dl_entry<i32, dense<32> : vector<2xi64>>, #dlti.dl_entry<f16, dense<16> : vector<2xi64>>, #dlti.dl_entry<f128, dense<128> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi64>>, #dlti.dl_entry<f64, dense<64> : vector<2xi64>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi64>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi64>>, #dlti.dl_entry<i64, dense<64> : vector<2xi64>>, #dlti.dl_entry<i128, dense<128> : vector<2xi64>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i64>, #dlti.dl_entry<"dlti.endianness", "little">>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @sub_(%arg0: tensor<?x?xf32> {fir.bindc_name = "a"}, %arg1: tensor<?x?xf32> {fir.bindc_name = "b"}) {
    //%0 = fir.alloca !fir.box<!fir.array<?x?xf32>>
    %c1 = arith.constant 1 : index
    %c1_i32 = arith.constant 1 : i32
    %c0 = arith.constant 0 : index
    %c4_i64 = arith.constant 4 : i64
    %c3_i64 = arith.constant 3 : i64
    %c2_i64 = arith.constant 2 : i64
    %c1_i64 = arith.constant 1 : i64
    //%1 = fir.dummy_scope : !fir.dscope
    %2 = arith.index_cast %c1_i64 : i64 to index
    %3 = arith.index_cast %c2_i64 : i64 to index
    //%4 = fir.shift %2, %3 : (index, index) -> !fir.shift<2>
    //%5 = fir.declare %arg0(%4) dummy_scope %1 {uniq_name = "_QFsubEa"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
    //%6 = fir.rebox %5(%4) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
    %7 = arith.index_cast %c3_i64 : i64 to index
    %8 = arith.index_cast %c4_i64 : i64 to index
    //%9 = fir.shift %7, %8 : (index, index) -> !fir.shift<2>
    //%10 = fir.declare %arg1(%9) dummy_scope %1 {uniq_name = "_QFsubEb"} : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.dscope) -> !fir.box<!fir.array<?x?xf32>>
    //%11 = fir.rebox %10(%9) : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>) -> !fir.box<!fir.array<?x?xf32>>
    %12 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %13 = arith.index_cast %12 : index to i64
    %14 = arith.trunci %13 : i64 to i32
    %15 = tensor.dim %arg1, %c0 : tensor<?x?xf32>
    %16 = arith.cmpi eq, %15, %c0 : index
    %17 = arith.index_cast %c1_i32 : i32 to index
    %18 = arith.select %16, %17, %7 : index
    %19 = arith.index_cast %18 : index to i32
    %20 = arith.addi %14, %19 : i32
    %21 = arith.subi %20, %c1_i32 : i32
    %22 = arith.extsi %21 : i32 to i64
    %23 = arith.index_cast %22 : i64 to index
    %24 = arith.subi %23, %7 : index
    %25 = arith.addi %24, %c1 : index
    %26 = arith.cmpi sgt, %25, %c0 : index
    %27 = arith.select %26, %25, %c0 : index
    %28 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %29 = arith.index_cast %28 : index to i64
    %30 = arith.trunci %29 : i64 to i32
    %31 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
    %32 = arith.cmpi eq, %31, %c0 : index
    %33 = arith.index_cast %c1_i32 : i32 to index
    %34 = arith.select %32, %33, %8 : index
    %35 = arith.index_cast %34 : index to i32
    %36 = arith.addi %30, %35 : i32
    %37 = arith.subi %36, %c1_i32 : i32
    %38 = arith.extsi %37 : i32 to i64
    %39 = arith.index_cast %38 : i64 to index
    %40 = arith.subi %39, %8 : index
    %41 = arith.addi %40, %c1 : index
    %42 = arith.cmpi sgt, %41, %c0 : index
    %43 = arith.select %42, %41, %c0 : index
    //%44 = fir.shape %27, %43 : (index, index) -> !fir.shape<2>
    //%45 = fir.undefined index
    //%46 = fir.slice %7, %23, %c1, %8, %39, %c1 : (index, index, index, index, index, index) -> !fir.slice<2>
    //%47 = fir.rebox %11(%9) [%46] : (!fir.box<!fir.array<?x?xf32>>, !fir.shift<2>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xf32>>
    %buf = tensor.empty(%12, %28) : tensor<?x?xf32>
    %buf_init = tensor.insert_slice %arg0 into %buf[0, 0][%12, %28][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    %48 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
    %49 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
    %tmp = tensor.extract_slice %arg1[0, 0][%15, %31][1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
//  %tmp = tensor.generate %48, %49 {
//  ^bb0(%arg2: index, %arg3: index)
//    %elem = tensor.extract %arg1[%arg2, %arg3] : tensor<?x?xf32>
//    tensor.yield %elem : f32
//  } : tensor<?x?xf32>
    %ret = tensor.insert_slice %tmp into %arg0[0, 0][%12, %28][1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
    return
  }
}

```

### 解説
* このお題を設定した意図はtensor dialect(とlinalg dialect)の検証だったのですが、引数を通して`tensor`をやりとりする術がないことがとりあえず分かりました
  * `tensor.insert_slice`はtensorをコピーしてそこに要素を挿入するため、`tensor.store`のような操作がないと`arg0`のtensorに反映されません。
  * `bufferization.to_memref`と`memref.copy`の組み合わせも考えましたが、`bufferization.to_memref`は型変換ではなく、新たにメモリ領域を確保してそこにコピーしているので同じ帰結になります。
  * `linalg.copy`を使うことも考えましたが、やはり同じ帰結になります。

### 考察
引数の`tensor`をbufferizeする機能が試験的な実装に留まっていることと関係があるのかなと思いました。
将来的にこの辺りが整備されてくれば、`tensor`を使って配列を表現する手段が確立できるかもしれません。

# 関連研究
ちょうど検証をしていた時期に以下の論文が発表されていました。
みんな考えることは同じですね。
(なのでこの記事は完全に二番煎じになってしまいました。)

[Fully integrating the Flang Fortran compiler with standard MLIR](https://arxiv.org/abs/2409.18824)

## 内容

* FlangはHLFIR->FIR->LLVM IRとLoweringしている
  * FIRからMLIR付属のdialectを経由せずいきなりLLVM IRに変換している
    * これではMLIRの最適化を活用できない
  * さらに言えばFlangの最適化をMLIRに還元することもない
* HLFIRに変換したあと、xDSLで独自実装したMLIR付属のdialectに変換するパスを通し、最後にMLIRの変換パスを使ってLLVM IRを生成した
  * `fir.if`や`fir.do_loop`はそれぞれ`scf.if`と`scf.for`にそれぞれ置き換える
  * `cf.br`での分岐先のブロックが未変換のものを指さないようにワンクッション置いて変換する
  * `fir.alloca`、`fir.load`、`fir.store`はそれぞれ`memref.alloca`、`memref.load`、`memref.store`に置き換える
    * `func.func`に`AutomaticAllocationScope`というTraitがあるにも拘らず、スタック上に確保された領域が適切に解放されていなかったので、`memref.alloca_scope`で関数呼び出しを囲ってスコープを明示的に指定するようにした
  * `fir.allocmem`と`fir.freemem`もそれぞれ`memref.alloc`と`memref.dealloc`に置き換える
    * 再割付けができるように2重のmemrefにする
      * そのままだと余計な処理が増えて性能が落ちるので、最適化パス(LICM?)を作ってデリファレンスの回数を抑えるようにした
  * グローバル配列は`memref.global`で、グローバルなスカラ変数は`llvm.mlir.global`で定義した
  * 一部の組込み関数には対応するlinalg dialectの表現があるので、それに置き換える
  * 派生型は成分を全部バラして管理するようにした
    * `tuple`型はまともに使えないので
* この変換処理を通した結果、性能向上に一定の効果があった
  * 概要には最大3倍と書いてあったが、結果の表を見る限りだと最大2.37倍
  * 要素ごとに見ていくと線形代数処理、OpenMPによる並列化、GPUオフロードで性能が大きく変わることが分かった
    * `DOT_PRODUCT`の性能が3.33倍向上
  * つまり既存のFlangの変換処理は実行効率の悪いものになっているということが言えるので、開発コミュニティはこの結果を真剣に受け止めるべき

## 気になったところ
概ね共感していますが、主に言語仕様の観点で気になったところがあったので書いておきます。

* 引数の型を`!fir.ref<i32>`から`i32`に変えている
  * Fortranでは基本的に引数は参照渡しされます。`ref`とはreferenceの略であり、LLVM IRに変換されるタイミングで`ptr`になります。一方で`i32`とすると値渡しになります。Fortranでも値渡しは出来ますが、ソースコード上で仮引数に`VALUE`属性を明示的に指定する必要があります。つまり、考えようによってはベンチマークプログラムを勝手に書き換えたことになるのですが、大丈夫でしょうか？
* `scf.for`の使い方がFortranの言語規格に沿っていない
  * これは先述の通り。
* `fir.iterate_while`はEXIT文やGOTO文を含むループのためのものではない
  * EXIT文やGOTO文を含むループはFlangでは"unstructured loop"という括りにされ、cf dialectを使った形に書き換えられます。そもそも`fir.iterate_while`では途中でループイタレーションの処理を途中で切り上げるといったことはできないはずです。
  * `fir.iterate_while`を`scf.while`に変換すること自体は問題ないと思います。
* スタック上の領域が解放されていないとは
  * スタックポインタを弄ることで領域を解放すると思うのですが、そのあたりのコードは~~LLVM IR上には現れず~~、機械語に変換するときにターゲットの呼び出し規約に基づいてLLVMのバックエンドが生成しています。
    * 今までちゃんと理解できていなかったのですが、LLVM IRのintrinsicである`llvm.stacksave`と`llvm.stackrestore`は関数の途中で領域を確保/解放したい場合に使われるみたいですね。([参考](https://rhysd.hatenablog.com/entry/2017/03/13/230119))
    * `memref.alloca_scope`で囲うと、そこが`llvm.stacksave`と`llvm.stackrestore`で挟まれるようになります。
    * Fortranでは(C89と同様に)宣言文を初めにまとめて書くことになっているので、明示的に解放処理を書かなくても大丈夫だと思いましたが、うまくいかないケースがあったのでしょうか？
      * 例外としてBLOCK文がありますが、特に言及されてないですね。
* 割付け変数を2重の`memref`で表現している
  * 言われてみれば自分は割付け変数のこと何も検証してなかったですが、再割付けは`memref.realloc`ではダメなんでしょうかね？
  * 既に確定した値を持っている割付け変数が再割付けされるとそれらの値がどうなるのか次第だと思います。(未確認)
  * 今のMLIRにはポインタ変数を扱う術がないので苦しいですね。

# おわりに
いかがでしたか？

今回はFortranプログラムをMLIR付属のdialectだけで表現してみました。
Fortranの配列表現は`memref.subview`を活用することで実現できました。
一方で、文字型や派生型、リテラルなどの表現はMLIR付属のdialectでは不十分なことが分かりました。
あとはaffine dialectと言語規格の間で板挟みになるのも課題ですね。

今回検証できたのは言語機能のごく一部でしかないので、もっといろいろ検証してみたいです。
