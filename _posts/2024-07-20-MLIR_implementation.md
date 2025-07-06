---
layout: post
title:  "MLIRの実装"
date:   2024-07-20 13:55:54 +0900
categories: コンパイラ
tag: mlir
---
# 導入
[前回](../../../2023/09/09/MLIR_for_beginners.html)はMLIRの概念的な部分の説明に徹して、実装部分の話はほとんどしなかった。

今回はいよいよ実装部分に踏み込む。
とはいえ、あまり深追いはしない。
ここではDialectを作成、拡張する上で必要になる知識をまとめる。

結構雑に書いているので、この記事の基になっている公式の[チュートリアル](https://mlir.llvm.org/docs/Tutorials/)も適宜参照するとよいと思う。

# 実装方法
## Dialect
まずはDialect自身を定義する。

MLIRはそれ自身が中間言語というわけではなく、中間言語を定義するためのフレームワークでしかない。
そのため、Dialectを実装しないことには始まらない。
(MLIRのプロジェクトの中で実装されているDialectも多数あるが、例えばFIRはMLIRのプロジェクトから外れたところにいる)

ベースとなるクラスはMLIRの中に定義されており、[mlir/include/mlir/IR/Dialect.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Dialect.h)に実装されている。
この`mlir::Dialect`を継承してDialectを定義する。

実はC++で直接定義する以外にもう一つ定義する方法がある。
それが[TableGen](https://llvm.org/docs/TableGen/ProgRef.html)を使った方法で、これをMLIRではODS(Operation Definition Specification)と呼んでいる。
ODSで定義する場合は、[mlir/include/mlir/IR/DialectBase.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/DialectBase.td)にある`Dialect`を継承して定義する。
TableGenは独自の文法からC++コードを生成する仕組みだが、それは`mlir-tblgen -gen-dialect-decls`というコマンドで実行される。  
また、ODSの別の利点として、ドキュメント化が楽というのがある。(-gen-dialect-doc)
そのため、ODSでの定義が推奨されている。以降でもODSでの定義方法を中心に説明する。

メンバとして定義できるものをいくつかピックアップする。

|メンバ|必要性|説明|(C++コードとの対応)|
|---|---|---|---|
|`name`|必須|Dialectの名前そのものズバリ|`getDialectNamespace()`|
|`cppNamespace`|推奨|C++コードでDialect固有の要素を定義するときに使う名前空間<br>(省略した場合は`name`と同じになる)|`namespace {}`|
|`summary`|推奨|Dialectの概要の説明||
|`description`|推奨|Dialectの具体的な説明||
|`dependentDialects`|適宜|定義するDialectに必要な既存のDialectのリスト||

それ以外のメンバについては、適宜`DialectBase.td`を参照して欲しい。

Dialectは定義するだけではダメで、実際に使用するには`MLIRContext`に`loadDialect()`で読み込ませる必要がある。(どのタイミングで？)

### Op (Operation)
Dialectが定義できたら中身を実装していく。
まずはIRの基本構成要素であるOperationを定義する。

DialectのOperationを定義するには、[mlir/include/mlir/IR/OpDefinition.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpDefinition.h)にある`mlir::Op`を継承してクラスを定義すればよい。
`mlir::Op`はCRTP(Curiously Reccuring Template Pattern)というもので、詳細はググってほしいがクラスを継承する際にテンプレート引数にサブクラス自身を渡す手法が使われている。  
(ちなみに`Operation`というクラスもあるが、これはIRの構成要素としてのOperationであって似て非なるものである。詳細は[公式チュートリアル](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/#op-vs-operation-using-mlir-operations)を参照。)

ODSでも定義できるが、いきなり[mlir/include/mlir/IR/OpBase.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpBase.td)にある`Op`を継承して定義するのはあまり推奨されていなさそう。
ODSではまずDialectのOperationを定義する`Op`のサブクラスを定義して、さらにそれを継承してOperationを定義するというのがベストプラクティスとされている。
生成コマンドは`mlir-tblgen -gen-op-decls`(クラス定義)と`mlir-tblgen -gen-op-defs`(関数定義)である。

定義できるものとして以下がある。

|メンバ|必要性|説明|(C++コードとの対応)|
|---|---|---|---|
|`mnemonic`(テンプレート引数)|必須|Operationの名前|`getOperationName()`|
|`traits`(テンプレート引数)|任意|Operationの性質|対応する`OpTrait`|
|`arguments`|必須|Operationの引数(OperandとAttribute)<br>なければ省略可|`mlir::OpTrait::ZeroOperands`など|
|`results`|必須|Operationの返り値<br>なければ省略可|`mlir::OpTrait::ZeroResults`など|
|`summary`|推奨|Operationの概要の説明||
|`description`|推奨|Operationの具体的な説明||
|`hasVerifier`|推奨|生成されたOperationの妥当性を確認する関数`verify()`をユーザが定義するか|`verify()`|
|`builders`|任意|コンストラクタの追加|`build()`|
|`assemblyFormat`<br>`hasCustomAssemblyFormat`|任意|アセンブリ形式での表現|`print()`および`parse()`|

その他、`region`や`successors`で分岐処理が作れる(よく分かっていない)ほか、`OpBase.td`にいくつかあるので適宜参照。

Operationも定義するだけではダメで、使用するためには先ほど定義したDialectのクラスの`initialize()`内で`addOperations()`を呼ぶ必要がある。  
このとき、`GET_OP_LIST`というマクロを定義した上で`-gen-op-defs`で生成されたファイルをインクルードすると簡潔に書ける。([参考](https://note.com/lewuathe/n/na64b95954988))

### Type

### Interface
一番使用頻度が高いのはOperationのInterfaceだと思うのでそれを中心に説明する。

OperationのInterfaceは[mlir/include/mlir/IR/OpDefinition.h](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/OpDefinition.h)にある`OpInterface`を継承して定義する。
そして定義したInterfaceにinterface methodを実装する。
(Op側にinterface methodを宣言する必要があるが、Interfaceのクラスを継承させればよい？後述するODSでのやり方しか情報がないため不明。
多分tblgenで自動生成されるファイルの中身を見ればわかるとは思うが)

ODSで定義する場合は、[mlir/include/mlir/IR/Interfaces.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/IR/Interfaces.td)の`OpInterface`を継承して定義する。
interface methodは`method`というメンバの中に列挙していく。
各interface methodは`InterfaceMethod`というテンプレートを使って宣言を書く。
最後にOperationからinterface methodを呼べるように、Traitsに`DeclareOpInterfaceMethods`をつけてInterfaceを指定すると、`Op`クラスのメンバとして必要な関数の宣言が自動で挿入される。(ただし`defaultImplementation`を指定したmethodについては明示的に指定しないとオーバーライドできない)
あとはそれらに対して定義を書けばよい。

DialectInterfaceを一から定義する方法は不明だが、定義したDialectInterfaceを使用するためにはDialectのクラスの`initialize()`内で`addInterfaces()`を呼ぶ必要がある。

### Pass
DialectにはOperationが定義されれば十分かといわれるとそうではない。
Dialectによる中間表現は、表現できることだけでなく、LLVM dialect、そしてLLVM IRに変換されていくことが求められる。
多くのDialectでは`IR`というディレクトリにOperationの定義、`Transforms`というディレクトリにOperationの変換規則(Pass)の定義がされている。
ここではそのPassを扱う。

ちなみにMLIRではLoweringパスもOptimizationパスも等しく"変換パス"という扱いになっている。
これは前回も述べた通りPartial Loweringが可能であるため、そもそもLoweringのフェーズというものが存在しないからである。
(ただし実装の観点では、インターフェースこそ共通だが中身の実装は明らかに違っている)

まずはパス自身を定義する。
パス自身は`PassWrapper`というクラスを使って定義する。
`PassWrapper`もCRTPであり、一番目の引数に定義するパス自身、二番目の引数には継承するクラス(`OperationPass<mlir::ModuleOp>`など)を指定する。
そしてメンバ関数である`runOnOperation()`などに具体的な処理内容を記述していく。

パスに関しても、C++で直接定義する以外にODSで定義する方法がある。
[mlir/include/mlir/Pass/PassBase.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Pass/PassBase.td)にある`Pass`を継承することで定義できる。
ただ、結局ODSはガワしか作ってくれないので中身は自分で実装していく必要がある。

次に`runOnOperation()`の中に変換処理を実装するにあたって役に立つ機能をいくつか紹介する。

まず1つ目は`getOperation()`である。  
これはパスの変換対象となるOperationを返してくれる関数である。
この関数はパスの中であればどこでも呼び出すことができる。
(当たり前と思われるかもしれないが、LLVMだと変換対象のInstructionを取得しようと思ったときに、そこから見えているクラスから辿っていかないと取得できないことがあり、面倒くさい)

2つ目は`mlir::RewritePattern`である。  
実際には`mlir::OpRewritePattern`や`mlir::OpInterfaceRewritePattern`、`mlir::ConversionPattern`などを継承して使う。
その名から分かるように、パターンマッチングでOperationを書き換えていく仕組みである。
変換処理の本体は`matchAndRewrite()`であり、パターンマッチングをした後`mlir::PatternRewriter`の`replaceOp()`によって実際に変換する。  
この時のパターンマッチをC++で書く方法もあるが、簡単なパターンマッチであればTableGenを使って簡潔に書くことができる。
MLIRではこれをDRR(Declarative Rewrite Rule)と呼んでいる。

3つ目は`mlir::ConversionTarget`である。  
大抵の場合、変換パスでは変換対象となるものとならないものがあるわけで、そのあたりの区別はちゃんとする必要がある。
特にDialect間の変換ではこの辺りを確認するコードを一から書いているのでは大変だし漏れがあるかもしれない。
そこでこの仕組みを使う。
(Conversionという単語自体はDialect間の変換に限らずDialect内での変換も含むはずだが、`ConversionTarget`はDialect間の変換(Lowering)に使われることがほとんどのようだ)  
この`ConversionTarget`に、`addLegalDialect`, `addDynamicallyLegalDialect`, `addIllegalDialect`, `addLegalOp`, `addDynamicallyLegalOp`, `addIllegalOp`といった関数で情報を追加していく。
例えば`addLegalDialect`と`addIllegalDialect`を組み合わせることで`addIllegalDialect`のOperationをすべて`addLegalDialect`に変換するといったことを表現できる。
また、`addIllegalDialect`のあるOperationを`addLegalOp`に追加することで、例外的にそのOperationが変換されなくても許されるようになる。  
あとは`RewritePatternSet`に`ConversionPattern`を追加し、`mlir::applyPartialConversion()`または`mlir::applyFullConversion()`を呼ぶことで変換が実行される。
(ちなみによく使われる変換パターンは既にまとめられていて、例えば`mlir::arith::populateArithToLLVMConversionPatterns()`といった関数を呼べばArith DialectからLLVM Dialectへの`RewritePattern`を一括取得できる。)

パスもやはり定義するだけではダメで、`mlir::PassManager`に`addPass()`でパスを追加した上で`run()`を呼んで初めてパスが実行される。  
このとき、`addPass()`に渡すパス(への`unique_ptr`)を生成する必要が出てくるわけで、そのための関数(`create*Pass()`)も必要になる。
中でやっていることは大したことはなく、ODSで勝手に作ってくれる。
(ODSで`constructor`を明に定義しておけばユーザ好みにカスタマイズできる)

## IR
Dialectを定義する方法を前節で述べた。ただこれだけの情報ではパスの中身は実装できないと思う。
ここからはMLIRの構造の実装を見ていく。

### 構造
LLVM IRの場合は、Module->Function->BasicBlock->Instructionと階層がはっきり分かれている。
対してMLIRの場合は、前回説明した通りOperationを中心として、Operation自身が階層構造を持つようになっている。
ModuleもFunctionもInstructionも、MLIRにおいては等しくOperationである。

[2020-02-26 - CGO 2020 Talk](https://docs.google.com/presentation/d/11-VjSNNNJoRhPlLxFgvtb909it1WNdxTnQFipryfAPU/edit#slide=id.g7d334b12e5_0_258)より図を拝借して、まずは各階層へのアクセスの仕方を確認する。

![MLIRの構造]({{site.baseurl}}/assets/img/2024-07-20/MLIR_Structure.png)

___

Operation->親Block: `getBlock`  
Block->親Region: `getParent`  
Operation->親Region: `getParentRegion`  
Region->親Region: `getParentRegion`  
Operation->親Operation: `getParentOp`  
Block->親Operation: `getParentOp`  
Region->親Operaion: `getParentOp`

Region->Block: `getBlocks`  
Region->BlockArgument: `getArguments`  
Region->Operation: `getOps`

Block->BlockArgument: `getArguments`  
Block->Operation: `getOperations`

Operation->OpOperand: `getOperands`  
Operation->Attribute: `getAttr`

Value->定義元Operation: `getDefiningOp`  
Value->親Block: `getParentBlock`  
Value->親Region: `getParentRegion`

Operation->その結果を使っているOpOperand: `getUses`  
Value->それを使っている(対応する)OpOperand: `getUses`  
Operation->その結果を使っているOperation: `getUsers`

OpOperand->親Operation: `getOwner`  
OpOperand->親Value: `get`

___

`Operation*`は`mlir::dyn_cast`を使って前項で定義したような任意の`Op`のサブクラスに変換できる。
また、OperationがあるInterfaceを持っているかどうかは、該当Interfaceのクラスに`dyn_cast`できるかで判別できる。

### Walker
例えばModuleからIR全体を探索して各Operationに対して同じ処理を行いたいとなったときに、便利な機能がある。
それがWalkerである。
正確に言えば`Operation::walk()`である。

この関数は引数に関数を取ることができ、そのOperationを起点に順次Operationを探索していき関数を適用していく。
デフォルトでは後行順の深さ優先探索(葉から辿る)をするが、テンプレート引数で設定すれば先行順で(根から)辿る。

https://mlir.llvm.org/docs/Tutorials/UnderstandingTheIRStructure/#walkers
