---
layout: post
title:  "LLVMの最適化メッセージ出力の仕組み"
date:   2025-12-20 13:02:51 +0900
categories: コンパイラ
tag: llvm clang mlir
---
## 導入
プログラムの性能分析やチューニングにおいて、コンパイラが既にどのような最適化をしているのか、どのような最適化が出来ていないのか、理由は何かを把握することは重要である。
GCCでは`-fopt-info`で、Clangでは`-Rpass=`/`-Rpass-missed=`/`-Rpass-analysis=`で最適化情報を出力してくれる。
我々が使う上で仕組みを知っていなければ使いこなせないといったことは全くないのだが、興味本位で調べてみた。

## 仕様
https://llvm.org/docs/Remarks.html

## LLVM
LLVMの最適化パスを眺めていると以下のような呪文が書かれているのを時折見かけると思う。
まずはこれの意味するところから確認しよう。

```cpp
// OptimizationRemarkEmitter ORE(L.getHeader()->getParent());
OptimizationRemarkEmitter &ORE = AM.getResult<OptimizationRemarkEmitterAnalysis>(F);
  ...
ORE.emit([&]() {
  return OptimizationRemark("my-opt-pass", "Accelerated", Inst)
         << "My special optimization is performed.";
});
```

最初の`OptimizationRemarkEmitter`は、`LoopPass`の場合はコンストラクタを呼び、`FunctionPass`の場合はOptimizationRemarkEmitterAnalysisから引っ張ってくるらしい。

そして本題の次の文。
まず、一番外には`OptimizationRemarkEmitter::emit`という関数があり、引数にラムダ式を取っている。
そしてこのラムダ式は`OptimizationRemark`に`<<`演算子を適用したものを返す。  
では内側から外側へ処理を辿っていく。
まず、`OptimizationRemark`のコンストラクタだが、種類がいくつかある。
詳細は[`llvm/include/llvm/IR/DiagnosticInfo.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/DiagnosticInfo.h#L766-L783)を確認されたい。  
次に`<<`演算子であるが、これは実態としては[`DiagnosticInfoOptimizationBase::insert`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/DiagnosticInfo.cpp#L432-L438)というメンバ関数が呼ばれている。
`DiagnosticInfoOptimizationBase`は`Args`という文字列のベクトルを持っていて、`insert`関数で追加していく。
この式は`OptimizationRemark`として評価されるため、このラムダ式は`OptimizationRemark`を返す関数ということになる。  
最後に`OptimizationRemarkEmitter::emit`を見る。
[`llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h#L77-L92)にあるように、`DiagnosticInfoOptimizationBase`への参照を渡す関数を受け取り、それを同名関数に渡して処理している。
中では、[`LLVMContext::diagnose`](https://github.com/llvm/llvm-project/blob/main/llvm/lib/IR/LLVMContext.cpp#L249-L273)を呼んで文字列を出力する。
なお、`DiagnosticInfoOptimizationBase`はメンバに`Function`を持っているため、そこから`LLVMContext`にアクセスできる。  

ここからは最適化に限らないメッセージ出力の仕組みとなる。
出力のパターンもいくつか存在する。
まず1つ目は、最適化関連(`DiagnosticInfoOptimizationBase`)であり尚且つ`LLVMContext`が`LLVMRemarkStreamer`を持っている場合である。
例えば`-fsave-optimization-record`を指定してYAMLファイルに情報を書き出すときが挙げられる。
この場合、`LLVMRemarkStreamer::emit`に`DiagnosticInfoOptimizationBase`を渡して出力させる。  
2つ目は、`LLVMContextImpl`が持つ`DiagnosticHandler`を使う場合である。
例えば`-Rpass=`/`-Rpass-missed=`/`-Rpass-analysis=`を指定したときが挙げられる。
`DiagnosticHandler`を使うとパターンマッチングによるフィルタリングが可能になる。
(詳細は後述するが)`DiagnosticHandler::handleDiagnostics`に`DiagnosticInfo`を渡して出力させる。  
3つ目はそれ以外の場合である。
おそらく最適化以外のメッセージだと思われる。
[`DiagnosticSeverity`](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/DiagnosticInfo.h#L50-L57)とメッセージ内容を標準エラー出力に吐かせる。

呼び出し関係をまとめると以下。

```
OptimizationRemarkEmitter::emit @ llvm/include/llvm/Analysis/OptimizationRemarkEmitter.h
-> operator<< @ llvm/include/llvm/IR/DiagnosticInfo.h
  -> DiagnosticInfoOptimizationBase::insert @ llvm/lib/IR/DiagnosticInfo.cpp
-> LLVMContext::diagnose @ llvm/lib/IR/LLVMContext.cpp
  -> DiagnosticHandler::handleDiagnostics
```

## Clang
さて、LLVMのメッセージ出力のためのインフラストラクチャは理解できた。
ただ、これが全てではない。
Cソースを渡したときとLLVM IRを渡したときで最適化メッセージの出方が違うことに気づくはずだ。
つまり、Cソースに対してはメッセージの出力内容をアレンジしているということになる。

実はClangは`DiagnosticHandler`の派生クラスを作り、`DiagnosticHandler::handleDiagnostics`をオーバーライドしている。
定義は[clang/lib/CodeGen/CodeGenAction.cpp](https://github.com/llvm/llvm-project/blob/main/clang/lib/CodeGen/CodeGenAction.cpp#L69)にある。


```
ClangDiagnosticHandler::handleDiagnostics @ clang/lib/CodeGen/CodeGenAction.cpp
  -> BackendConsumer::DiagnosticHandlerImpl @ clang/lib/CodeGen/CodeGenAction.cpp
    -> BackendConsumer::OptimizationRemarkHandler @ clang/lib/CodeGen/CodeGenAction.cpp
      -> BackendConsumer::EmitOptimizationMessage @ clang/lib/CodeGen/CodeGenAction.cpp
        -> DiagnosticsEngine::Report @ clang/lib/Basic/Diagnostic.cpp
```

Clangのメッセージ出力機能は最適化だけでなく、構文解析や意味解析でも使われる仕組みだが、それに関してはまた別の機会にまとめることにしたい。

https://clang.llvm.org/docs/InternalsManual.html#the-diagnostics-subsystem

## MLIR(番外編)


## まとめ

