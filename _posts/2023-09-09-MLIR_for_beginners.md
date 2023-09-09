---
layout: post
title:  "MLIR入門"
date:   2023-09-09 11:37:16 +0900
categories: コンパイラ
tag: mlir
---
# 導入
最近Flangを見ているのだが、Flangフロントエンドで扱う内部表現はMLIRをベースにしたFIRというものが使われているらしい。
そこで今回はまずMLIRについて見ていきたい。
(具体的な実装の話はしない)

# MLIRとは
## 背景・概要

MLIRは、元々はTensorFlow向けに立ち上げられたGoogle発のプロジェクトであるが、その適用範囲は機械学習に限らない。
MLIRで表現できるものとして、SSA形式のIRの他、ASTやターゲット固有の命令、高位合成における回路などが挙げられている。
これらはDialectという仕組みを用いて各々の自由にMLIRを拡張することで実現できる。

[MLIR がもたらす新たな中間表現の可能性](https://zenn.dev/acd1034/articles/230423-mlir3vdt)

## 構造
[MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)

MLIRは基本的にグラフ構造を取っており、Operation(節点)とValue(枝)からなる。

Operationは階層構造を持つことができ、Operationの中にRegionを持ち、その中にBlockを持ち、その中にさらにOperationを持ち…といった構造を取りうる。
もちろんOperationが複数のRegionを持つことなども可能である。
そのため、Operationは抽象度の異なる様々な概念を扱うことができる。
(例えば関数定義はOperationで表現されるし、その中の演算もOperationで表現される。LLVM IRだと両者はFunctionとInstructionとして明確に区別される。)

Valueは型を持ち、ただ1つのOperationまたはBlock引数の結果を表す。

コード変形はPassで行われるが、上述のようにOperationは様々なものが表現できるため、これらをすべて考慮してコード変形を設計するのは厳しい。
そこでOperationにTraitとInterfaceという概念を持たせて制御する。

# MLIRの言語仕様
ここからは具体的にMLIRの言語仕様を見ていく。
基本的には内部的な実装の話ではなくMLIRの表現の話が主である。

## Value
Valueは接頭辞`%`が付いた識別子をもつものとして表現される。
ここでいう識別子とは、数値または数字以外から始まる文字列である。  
例: `%foo`, `%2`

また、Valueは配列にもなる。定義の時は`:`の後ろに大きさを書き、参照の時は`#`の後ろに添字(0-index)を書く。  
例: 
```
// 2つの結果を返すfooというOperationの結果を、大きさ2のresultというValueで受ける
%result:2 = "foo"() : () -> (f32, i32)

// resultの先頭の要素を、barというOperationの引数として渡す
"bar"(%result#0) : (f32) -> ()
```

## Operation
Operationで様々なものを表現できるようにいろいろな構成要素を持たせているせいで複雑なので、段階を追って説明する。

### 最小構成
最小構成のOperationは例えば以下である。

```
%foo, %bar = "foo_div"() : () -> (f32, i32)
```

* `%foo, %bar =`
    * Operationの結果を受けるValueのリスト。`,`で区切って並べる。
    * 結果を返さないOperationもあるので、ないこともある。(最小構成じゃないじゃん)
* `"foo_div"`
    * Operation名。ユニークであれば何でもいいらしいが、"`dialect`.`mnemonic`"の形で書かれることが多い。
* `()`(最初の括弧)
    * Operationの引数として渡すValue(Operand)のリスト。`,`で区切って並べる。
    * 引数がない場合でも括弧は省略できない。
* `: () -> (f32, i32)`
    * Operationの型情報。左が引数、右が結果。
    * 要素が1つしかない場合は括弧を省略できる。(要素がない場合は？)

### その他の構成要素
* 後続ブロックのリスト
    * `[]`で囲われたBlockのリスト。`,`で区切って並べる。
    * Blockの引数がある場合は、Block名の後ろに`:`をつけ引数を並べる。
    * 後述するが、条件分岐の表現としてΦノードの代わりに使われる。
* Attributeのリスト
    * `{}`で囲われたAttributeのリスト。`,`で区切って並べる。
    * Attribute自体は`entry` = `value`の形で書かれる。
        * Attributeもある意味Operationの引数である。
    * リストにも2種類あり、PropertyとしてOperationに保存されるものと、捨てられるものがある。
        * 前者はAttributeのリストをさらに`<>`で囲う必要がある。
* Regionのリスト
    * `()`で囲われたRegionのリスト。`,`で区切って並べる。
* ソースコード位置情報
    * デバッグのためにソース位置情報をつけている。
    * `loc("example.ll":9:17)`のような形式で表現される。

あとdialectごとに表現形式を自由に拡張できるらしい。  
(関数定義がこのルールに従ってないのはfunc dialectの*custom assembly form*だったりする？)

## Block
LLVM IRのBasicBlock相当。
違いは引数を取れることと、原則としてBlockの末尾がterminator operation(`cf.br`や`return`など)であること。

Blockはラベルと1つ以上のOperationからなる。
ラベルは、接頭辞`^`がついたBlock名の後に引数のリストを置き、末尾に`:`をつけたものである。
ちゃんとした例は[Reference](https://mlir.llvm.org/docs/LangRef/#blocks)を見てほしいが、以下のような感じである。

```
^bb0(%a: i64, %cond: i1):
  cf.cond_br %cond, ^bb1, ^bb2
```

## Region
Regionにも2種類あり、SSACFG regionとGraph regionがある。
両者の違いはBlock間の制御フロー(Operationの実行順序)が表現されているかどうかである。
(なんとなく言いたいことはわかるが具体的なGraph regionの例が思いつかない。)

Regionは`{}`の中にBlockを並べたものであり、Region自身は名前を持たない。  
Regionの先頭のBlockは特別に"entry block"と呼ばれ、
entry blockの引数はRegionの引数と一致するほか、entry blockはOperationの後続ブロックのリストに指定できない。
言い換えると、Regionに入るときは必ずentry blockから入り、出るときは適切なterminatorを持つ任意のBlockから出られる。(Single-Entry-Multiple-Exit (SEME) regions)  
なお、Graph regionは現状では(特段の理由はないが)単一のBlockしか持たない仕様になっている。

前述のとおりRegionは引数を取るほか、結果(空でもよい)を返す。
これらは親となるOperationと紐づけられている。

Operationは複数のRegionを取りうるが、Region間の制御の移り変わりは上記とは異なる。  
Operationは制御フローをどのRegionに渡してもよいし、同時に複数のRegionに渡すこともできる。
また、関数呼び出しなどで他のRegionに移ることもあり得る。

## Module
terminator operationを持たない単一のBlockを持つ単一のRegionをもつOperation。
(つまりModuleの中はOperationが並んでいるだけ)  
builtin dialectに定義されており、top-levelのOperationとしてIRを格納するのに使える。
パーサはパース結果を`ModuleOp`で返すことが期待される。

## 名前の有効範囲(スコープ)
Cのブロックと同じような感覚で変数のスコープはRegion内で閉じている。  
ただし例外があって、あるOperationについてオペランドの値を参照することが正当であるならば、そのOperationが持つRegionの中のOperationは(Regionの外で定義されたものにも関わらず)その値を参照できる。
これが気に入らない場合は、OpTrait::IsolatedFromAboveなどのTraitsやカスタムVerifierで制限できる。

ある値が同一Regionの他のOperationから参照できるかはRegionの種類で決まる。  
あるRegion内で定義された値は、そのRegion内に親を持つOperationから参照できる。(ただし親がその値を参照できる場合に限る。)
Regionの引数によって定義された値は、そのRegion内にあれば階層が深いOperationであってもその値を参照できる。  
反対に、あるRegion内で定義された値はその外側から参照できない。

また、同一Region内のBlockにしか分岐できない(terminator operationの引数に渡せない)。

## 型
MLIRにおける型は、組込みの型か型の別名(エイリアス)かdialectの型のいずれかである。

* 組込みの型
    * [builtin dialectの型](https://mlir.llvm.org/docs/Dialects/Builtin/#types)のこと。
* 型の別名
    * `!alias-name = type`の形で定義され、使うときは`!alias-name`。
* dialectの型
    * `!dialect<type>`または`!dialect.type`の形で使われる。

## Attribute
Attributeも型と同様、組込みのAttributeかAttributeの別名かdialectのAttributeのいずれかしかない。

* 組込みのAttribute
    * [builtin dialectのAttribute](https://mlir.llvm.org/docs/Dialects/Builtin/#attributes)のこと。
* Attributeの別名
    * `#alias-name = attribute`の形で定義され、使うときは`#alias-name`。
* dialectのAttribute
    * `#dialect<attribute>`または`#dialect.attribute`の形で使われる。

## Property
Operationのところでちょこっと書いたが、Operationに追加情報を持たせることができる。
これらはAttributeの形で表現できるほか、Interfaceのアクセサからも見れる。

### Interface
DialectレベルのInterfaceとAttribute/Operation/TypeのInterfaceがある。

いまいち使い方が分からない。

### Trait
いまいち使い方が分からない。

## Dialect
MLIRの拡張性を担っている概念である。  
これによって独自のOperation、Attribute、Typeを定義できる。
Dialectには名前空間が与えられて、Operationなどの接頭辞として使われる。

MLIRでは複数のDialectを使うことができる。
これは、1つのモジュールの中に複数のDialectが共存できるという意味である。

また、MLIRではDialectの変換のためのフレームワークが用意されている。
特徴的なのがPartial Conversion(Partial Lowering)である。
これは部分的にDialectを変換していく仕組みであり、上述の仕様をうまく利用したものとなっている。(因果関係が逆かもしれない。)

### Dialectの例
[https://mlir.llvm.org/docs/Dialects/](https://mlir.llvm.org/docs/Dialects/)
* Builtin
* llvm
* func
* arith
* affine
* linalg
* vector
* tensor

など…

Dialectについては[MLIRでHello, Tensor](https://note.com/lewuathe/n/na64b95954988)という記事も参考になる。
