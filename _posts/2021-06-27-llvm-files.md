---
layout: post
title:  "LLVMのディレクトリ構成"
date:   2021-06-27 17:27:02 +0900
categories: コンパイラ
tag: llvm
---
## 導入
[前回](../../../2021/06/27/llvm-ir-0.html)書いた今後やるべきことリストを確認する。
- LLVM内部での処理の流れを追う
- LLVMのデータ構造を知る
- LLVMのファイル構成を知る
- ~~LLVM IRを知る~~

LLVM IRについては(本当に簡単にではあるが)学んだので、今回はファイル構造を知る。
どこのファイルにどんなものが入っているかを理解する。

これも[ドキュメント](https://llvm.org/docs/GettingStarted.html#directory-layout)に書いてあるみたいだけどね。

## llvm-project
[GitHub](https://github.com/llvm/llvm-project)から落としてきた場合は、LLVMプロジェクトの各プロジェクト用のディレクトリがいろいろ入っている。  
以下ではドキュメントに沿って`llvm`の中身を説明していく。(他のプロジェクトも似たような構成にはなっている)
### llvm
#### cmake
モジュール単位でのcmake用のファイルが置かれている

#### examples
LLVMの動作を理解するためのサンプルコードがある。

#### include
言わずもがなヘッダーファイルが入っている。量が多すぎる。  
あと`llvm`と`llvm-c`というのがあるが、基本的には`llvm`の方だけ見ればいいらしい。

[Doxygen](https://llvm.org/doxygen/files.html)とかも有効活用した方がいいかも。

#### lib
ソースコード(C++)が入っている。~~紛らわしいからsrcとかにしてほしい~~

いくつか重要そうなものをピックアップ
- IR: `Instruction`や`BasicBlock`といったクラスはここで扱う
- Analysis: 解析をする
- Transform: 最適化(コード変形)をする
- CodeGen: LLVM IRから実際のコードを生成する(命令の選択、スケジューリング、レジスタ割当てなど)

#### test
LLVMの機能テスト用のディレクトリ  
テストデータはLLVM IR

#### tools
便利なツール群のソースコード  
GNU Binutilsみたいな感じ

## まとめ
`lib`の中にC++ソースコードが、`include`の中にヘッダーファイルが入っている。  
量が多いので適宜[Doxygen](https://llvm.org/doxygen/files.html)を参照した方がよさそう。
