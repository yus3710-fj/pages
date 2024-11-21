---
layout: post
title:  "Fortranの言語機能(共通機能編)"
date:   2023-05-07 11:07:40 +0900
categories: プログラミング言語
tag: C C++ Fortran
---
# 導入
今までC言語をメインに触っていたが、ここに来てFortranをやることになったので勉強している。
ただ、日本語で書かれている情報が少ないため(全く無いわけではないが初学者向けではないと感じる)、
自分なりにちゃんとまとめてみようかと思う。

参考までに筆者はCはそれなりに、JavaとC++は多少使えるという感じだ。

今回は「C/C++(一部他言語を含む)でできたことをFortranでやるにはどうするか」という観点でまとめていきたい。

# 前提
## 言語規格について
Fortranは今もなお規格が改訂され続けている言語である。今でも使われる主な規格を列挙すると以下が挙げられる。
* FORTRAN 77
* Fortran 90
* Fortran 95
* Fortran 2003
* Fortran 2008
* Fortran 2018
* Fortran 2023(これを書いてる間に追加されたが今回は無視する)

ちなみにFORTRAN(全部大文字)ではなくFortranとなっているのがいわゆるModern Fortran[^Modern]である。

規格が改訂されるとともに古い規格が少しづつ廃止されたりリニューアルされたりしているため、比較の際にはFortranの規格ごとの差も載せるようにする。
その際、廃止事項や廃止予定事項はその時点で除外する。
(CやC++も規格が改訂されているのは同じだが今回は区別しない)

[^Modern]: Fortran 90はFORTRAN 77の仕様をそのまま引き継いでいることもあり、Modern FortranはFortran 2003以降とする考え方もある。([参考文献](https://qiita.com/cure_honey/items/e06b89e238c3df3df693))

## 表記について
* "FORTRAN 77"や"Fortran 90"は長いので"F77"や"F90"と略して表記することにする
* Fortranでは大文字/小文字を区別しないためどちらで書いてもよいのだが、ここでは予約語や固有名は全て大文字で表記することにする
* F77では固定形式だが、それを無視した書き方になっていることを予めご了承いただきたい
  * コメントも`!`で統一する
  * 固定形式自体F95で廃止予定事項になっている
* スペースの都合で変なところで改行されてしまうが、Fortranでは空白の有無は無視されることがあるが改行すると別の文として扱われるので注意

# 型
Fortranでは変数の定義にも関数の定義にもとにかく型が必要である。
まずはFortranでの型を見ていく。

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>整数型</td>
        <td><code>int</code></td>
        <td><code>INTEGER</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>単精度浮動小数点型</td>
        <td><code>float</code></td>
        <td><code>REAL</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>倍精度浮動小数点型</td>
        <td><code>double</code></td>
        <td><code>DOUBLE PRECISION</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)<br>または<br><code>REAL(REAL64)</code>※ISO_FORTRAN_ENV</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>論理型</td>
        <td><code>bool</code>※stdbool.h</td>
        <td><code>LOGICAL</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>複素数型</td>
        <td><code>float complex</code>※complex.h</td>
        <td><code>COMPLEX</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>文字型</td>
        <td><code>char</code></td>
        <td><code>CHARACTER</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>文字列型</td>
        <td><code>char [30]</code><br>または<br><code>std::string x(30)</code>※string</td>
        <td><code>CHARACTER*30</code></td>
        <td>(同左)<br>または<br><code>CHARACTER(30)</code></td>
        <td><code>CHARACTER(30)</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>構造体/クラス<br>(派生型)</td>
        <td>
<pre>class point {
  int x, y;
  double dist();
};</pre>
        </td>
        <td>
<pre>TYPE point
  INTEGER x, y
END TYPE</pre>
        </td>
        <td>(同左)</td>
        <td>
<pre>TYPE point
  INTEGER x, y
  CONTAINS
    PROCEDURE :: dist
END TYPE</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>共用体</td>
        <td>
<pre>union data {
  int x;
  double y;
};</pre>
        </td>
        <td>
<pre>
INTEGER x
DOUBLE PRECISION y
EQUIVALENCE(x, y)</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td></td>
    </tr>
</table>

* 倍精度浮動小数点型を`REAL*8`や`REAL(8)`としている文献もあるが、この8という数字(種別値)は必ずしもバイト数を指すわけではないので非推奨らしい([参考文献](https://www.nag-j.co.jp/fortran/FI_4.html))けどメジャーなコンパイラは全てバイト数になっているらしいので別にいいのではとも思う
  * F08以降では`ISO_FORTRAN_ENV`モジュールの種別値を使うことを推奨
* Fortranは文字と文字列の区別がない
  * ただし文字の配列と文字列は明確に区別する必要がある
* Fortranで派生型(構造体のこと)がメンバ関数を持てるようになったのはF03以降

## 属性(型修飾子)

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>定数</td>
        <td><code>const int x = 1;</code>
        </td>
        <td>
<pre>INTEGER x
PARAMETER (x = 1)</pre>
        </td>
        <td>(同左)<br>または<br><code>INTEGER, PARAMETER :: x = 1</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>ポインタ</td>
        <td><code>int *p;</code></td>
        <td></td>
        <td>
<pre>INTEGER x
POINTER x</pre>または<br><code>INTEGER, POINTER :: p</code>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>配列</td>
        <td><code>int x[10];</code></td>
        <td><code>INTEGER x(10)</code><br>または<br><pre>INTEGER x
DIMENSION x(10)</pre></td>
        <td>(同左)<br>または<br><code>INTEGER, DIMENSION(10) :: x</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>静的変数</td>
        <td><code>static int x;</code></td>
        <td><pre>INTEGER x
SAVE x</pre></td>
        <td>(同左)<br>または<br><code>INTEGER, SAVE :: x</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

配列、ポインタ、文字型、派生型についてはまた別の章でも詳しく見ていく。

# リテラル
Fortranの型が分かったところで次はリテラルを見ていく。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 整数リテラル | `1` | `1` | (同左) | (同左) | (同左) | (同左) |
| 長整数リテラル | `1L` | | `1_8` | (同左) | (同左)<br>または<br>`1_INT64` ※ISO_FORTRAN_ENV | (同左) |
| 単精度浮動小数点リテラル | `1.0f`, `1.0e3` | `1.0`, `1.0E3` | (同左) | (同左) | (同左) | (同左) |
| 倍精度浮動小数点リテラル | `1.0` | `1.0D0` | (同左) | (同左) | (同左)<br>または<br>`1.0_REAL64` ※ISO_FORTRAN_ENV | (同左) |
| 四倍精度浮動小数点リテラル | `1.0L` | | | | `1.0_REAL128` ※ISO_FORTRAN_ENV | (同左) |
| 文字列リテラル | `"Hello"` | `'Hello'` | (同左) | (同左) | (同左) | (同左) |
| 複素数リテラル | `1.0f + 2.0f * I`※complex.h | `(1.0, 2.0)` | (同左) | (同左) | (同左) | (同左) |
| 論理型リテラル | `true`, `false`※stdbool.h | `.TRUE.`, `.FALSE.` | (同左) | (同左) | (同左) | (同左) |

* Fortranは''と""の区別がない
* 先述のとおり種別値は必ずしもバイト数を指すわけではないため、`1_8`が64bit整数になるとは限らない
  * F77にはそもそも種別値という概念がない
* 四倍精度浮動小数点は、F08以前では拡張仕様だが`1.0Q0`や`1.0_16`と書くことができる場合がある

# 変数
次はいよいよ変数を見ていく。
ここでは個別具体的な事情には目をつぶり一般的な話だけ述べる。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 宣言 | `int x;` | `INTEGER x` | `INTEGER :: x` | `INTEGER x` | (同左) | (同左) |
| 代入 | `x = 1;` | `x = 1` | (同左) | (同左) | (同左) | (同左) |
| 初期化 | `int x = 1;` | `DATA x/1/` | (同左)<br>または<br>`INTEGER :: x = 1` | (同左) | (同左) | (同左) |

* 属性や初期化子がない場合は`::`を省略できる
  * F03以降は属性があっても省略できるようになった(?)

# 配列

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 配列要素 | `x[i][j]`<br>または<br>`*(x+i*N+j)` | `x(j,i)` | (同左) | (同左) | (同左) | (同左) |
| 配列の大きさ | `sizeof(x)/sizeof(x[0])` | | `SIZE(x)` | (同左) | (同左) | (同左) |
| 配列の初期化(一次元) | `int x[5] = {1,2,3,4,5};` | `DATA /x/1,2,3,4,5/` | (同左)<br>または<br>`x = (/1,2,3,4,5/)` | (同左)<br>または<br>`x = [INTEGER :: 1,2,3,4,5]` | (同左) | (同左) |
| 配列の初期化(多次元) | `int x[2][5] = {1,2,3,4,5,6,7,8,9,10};` | `DATA /x/1,2,3,4,5,6,7,8,9,10/` | (同左)<br>または<br>`x = RESHAPE((/1,2,3,4,5,6,7,8,9,10/),(/5,2/))` | (同左) | (同左) | (同左) |

* Cの配列は0-based indexingだが、Fortranは自由に決められる(デフォルトは1-based)
* Fortranの配列を用いた処理は簡潔に書けるものが多いが、それについてはCとFortranの違い(言語固有機能編)で触れる
* Fortranには"ポインタの配列"はない
  * 後述の配列(への)ポインタのみ
* Fortranの配列構成子`(/ /)`は1次元配列を作るので、そのまま多次元配列に代入しようとすると文法エラーになるため、`RESHAPE`関数で形状を変える必要がある

## 部分配列
C/C++にはないがPythonにはあるのでPythonと比較して説明する。

| 機能 | Python | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 部分配列の作成<br>(スライス) | `x[i:j:k]` | | `x(i:j-1:k)` | (同左) | (同左) | (同左) |

* Fortranにはさらにベクトル添字というものがあり、かなり自由に部分配列を作れる
  * 詳細はCとFortranの違い(言語固有機能編)で

## 文字列
Fortranにおける文字の配列と文字列の違いはCとFortranの違い(言語固有機能編)で触れることにして、
ここでは単に文字列の扱い方を触れておく

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 文字要素 | `x[i]` | `x(i:i)` | (同左) | (同左) | (同左) | (同左) |
| 部分文字列 | `x.substr(i,j)`※string | `x(i:i+j)` | (同左) | (同左) | (同左) | (同左) |
| 長さ取得 | `strlen(x)`※string.h<br>または<br>`x.size()`※string | `LEN(x)` | (同左) | (同左) | (同左) | (同左) |

* この文字要素の参照の仕方はまさしく(増分を指定できない)部分配列である
  * 長さが1でも`:`を省略できない点が特殊

## 動的割当
Fortranでは割付けという言い方をする。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 割当て | `x = malloc(sizeof(int)*N)` | | `ALLOCATE(x(N))` | (同左)<br>または<br>`ALLOCATE(INTEGER::x(N))` | (同左) | (同左) |
| 解放 | `free(x)` | | `DEALLOCATE(x)` | (同左) | (同左) | (同左) |

* `ALLOCATE`関数の引数には配列ポインタか割付け配列と呼ばれるものを指定する
  * このあたりはCとFortranの違い(言語固有機能編)でもう少し説明する
* Fortranでは明示的に解放しなくても各プログラムの終わりに到達すると(一部を除いて)勝手に解放されることになっている

# ポインタ
C/C++のポインタはアドレスを表現するものだが、Fortranでは単に別名(エイリアス)を与えるだけである。
(なので使用場面は限られる気がするのだがどうなんだろう)

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 配列へのポインタ型 | `int (*)[10]` | | `INTEGER, POINTER, DIMENSION(:)` | (同左) | (同左) | (同左) |
| ポインタ代入 | `p = &t` | | `p => t` | (同左) | (同左) | (同左) |
| NULL化 | `p = 0` | | `NULLIFY(p)` | (同左) | (同左) | (同左) |

* FortranのポインタはTARGET属性を持つものしか指せない
* C/C++では配列へのポインタは各次元の要素数が一致するものしか指せないが、Fortranでは自由
  * 次元数はどちらも一致している必要がある
* ちなみに`ISO_C_BINDING`の`C_LOC`関数や拡張仕様の`LOC`関数を使えば変数のアドレスは取得できる(Cでいうところの単項演算子`&`)。

# 派生型

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>継承(拡張)</td>
        <td><pre>class point {
  int x, y;
  virtual void output();
};
class point_3d : public point {
  int z;
  void output();
};</pre>
        </td>
        <td></td>
        <td></td>
        <td><pre>TYPE :: point
  INTEGER :: x, y
  CONTAINS
    PROCEDURE :: output
END TYPE
TYPE, EXTENDS(point) :: point_3d
  INTEGER :: z
  CONTAINS
    PROCEDURE :: output
END TYPE</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>多態性/多相性<br>(ポリモーフィズム)</td>
        <td><pre>point_3d var(1,2,3);
point &p = var;
p.output(); // (1, 2, 3)</pre>
        </td>
        <td></td>
        <td></td>
        <td><pre>TYPE(point_3d) :: var
CLASS(point), POINTER :: p => var
CALL output(p) ! (1, 2, 3)</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>抽象型</td>
        <td><pre>class animal {
  virtual void sound() = 0;
};</pre>
        </td>
        <td></td>
        <td></td>
        <td><pre>TYPE, ABSTRACT :: animal
  CONTAINS
    PROCEDURE :: sound
END TYPE</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

* Fortranの多相性の例に書いた実装を実現するには、`output`関数が引数を`CLASS(point)`として受け取り`SELECT TYPE`構文で型ごとに処理を分ける必要がある
  * C++のように継承の際に関数をオーバーライドして動作を分けることもできるが、型宣言の中に関数定義は書けないので上記の方法の方が結局分かりやすいか
  * 多態性とはインスタンスによって関数の動作を変えることなので実はこれで十分
* C++の場合、抽象クラスには純粋仮想関数が必要だが、Fortranには特に制約はない？

# 演算子
計算を行う上で欠かせないのが演算子である。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 和 | `a + b` | `a + b` | (同左) | (同左) | (同左) | (同左) |
| 差 | `a - b` | `a - b` | (同左) | (同左) | (同左) | (同左) |
| 積 | `a * b` | `a * b` | (同左) | (同左) | (同左) | (同左) |
| 商 | `a / b` | `a / b` | (同左) | (同左) | (同左) | (同左) |
| 余り | `a % b` | `MOD(a, b)` | (同左) | (同左) | (同左) | (同左) |
| 累乗 | `pow(a, b)` | `a ** b` | (同左) | (同左) | (同左) | (同左) |
| 小なり | `a < b` | `a .LT. b` | (同左)<br>または<br>`a < b` | (同左) | (同左) | (同左) |
| 小なりイコール | `a <= b` | `a .LE. b` | (同左)<br>または<br>`a <= b` | (同左) | (同左) | (同左) |
| 大なり | `a > b` | `a .GT. b` | (同左)<br>または<br>`a > b` | (同左) | (同左) | (同左) |
| 大なりイコール | `a >= b` | `a .GE. b` | (同左)<br>または<br>`a >= b` | (同左) | (同左) | (同左) |
| 等しい | `a == b` | `a .EQ. b` | (同左)<br>または<br>`a == b` | (同左) | (同左) | (同左) |
| 等しくない | `a != b` | `a .NE. b` | (同左)<br>または<br>`a /= b` | (同左) | (同左) | (同左) |
| 論理否定 | `!a` | `.NOT. a` | (同左) | (同左) | (同左) | (同左) |
| 論理積 | `a && b` | `a .AND. b` | (同左) | (同左) | (同左) | (同左) |
| 論理和 | `a || b` | `a .OR. b` | (同左) | (同左) | (同左) | (同左) |
| 排他的論理和 | `a ^ b`<br>※厳密にはビット演算子 | `a .NEQV. b` | (同左) | (同左) | (同左) | (同左) |
| 同値(XORの否定) | `a == b` | `a .EQV. b` | (同左) | (同左) | (同左) | (同左) |
| 文字列連結 | `a + b`※string | `a // b` | (同左) | (同左) | (同左) | (同左) |

* Fortranのビット演算はF90以降の組込み関数で提供される(`IAND`など)
  * 論理積などはあくまで論理型に対してのみ使えるもの
* 剰余はなぜか実数型に対しても使える

## ユーザ定義演算子
F90以降では`OPERATOR`文を使って演算子を再定義したり新しく定義することが可能である。
(C++では再定義だけのはず)

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>演算子のオーバーロード</td>
        <td><pre>point operator+(const point& lhs, const point& rhs) {
  pointer ret(lhs.x + rhs.x, lhs.y + rhs.y);
  return ret;
};</pre>
        </td>
        <td></td>
        <td><pre>INTERFACE OPERATOR(+)
  FUNCTION vadd(l, r)
    TYPE(point) :: l, r, vadd
  END FUNCTION
END INTERFACE
  :
FUNCTION vadd(l, r)
  TYPE(point) :: l, r, vadd
  vadd = (l%x + r%x, l%y + r%y)
END FUNCTION</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

# 条件分岐

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>if文</td>
        <td><pre>if (cond1) {
  y = -x;
} else if (cond2) {
  y = 0;
} else {
  y = x;
}</pre>
        </td>
        <td>
<pre>IF (cond1) THEN
  y = -x
ELSE IF (cond2) THEN
  y = 0
ELSE
  y = x
END IF</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>gotoを使ったループ</td>
        <td><pre>i = 0;
loop:
i++;
if (i <= N) goto loop;
</pre>
        </td>
        <td><pre>i = 0
10 CONTINUE
i = i + 1
IF (i .LE. N) GO TO 10</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>switch文</td>
        <td><pre>switch (a) {
  case 1:
    x = 1;
    break;
  case 2:
    x = 10;
    break;
  case 3:
    x = 11;
    break;
}</pre>
        </td>
        <td><pre>GO TO (10,20,30) a
10 x = 1
GO TO 100
20 x = 10
GO TO 100
30 x = 11
GO TO 100
100 CONTINUE</pre>
        </td>
        <td>(同左)<br>または<br><pre>SELECT CASE (a)
  CASE (1)
    x = 1
  CASE (2)
    x = 10
  CASE (3)
    x = 11
END SELECT</pre></td>
    </tr>
</table>

* Cにはif文しかないが、Fortranには論理IF文、ブロックIF文などがある。
  * この3つは文法的な細かな違いはあるが論理的な違いはあまりない
    * Fortranには他にもIF文があり、それらはなかなか癖がある
  * 論理IF文はGOTO文と組み合わせて使われるイメージだが、ループの中だとマスク付き演算と見ることも一応できる
* Fortranの`CONTINUE`はC/C++の`continue`とは全くの別物なので注意
  * C/C++でいう空(`;`のみ)の文
  * 文番号をつけるためのものという認識だがそれ以外に使い道ある？

# ループ
<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>forループ<br>(DOループ)</td>
        <td><pre>for (i = 0; i < N; i++) {
  a[i] = i;
}</pre>
        </td>
        <td>
<pre>DO 10 i = 1, N, 1
  a(i) = i
10 CONTINUE
</pre>
        </td>
        <td>(同左)<br>または<br><pre>DO i = 1, N, 1
  a(i) = i
END DO</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td><pre>DO i = 1, N, 1
  a(i) = i
END DO</pre>
        </td>
    </tr>
    <tr>
        <td>do-whileループ</td>
        <td><pre>do {
  i = i - 1;
} while(i > 0);</pre>
        </td>
        <td></td>
        <td>
<pre>i = i - 1
DO WHILE(i > 0)
  i = i - 1
END DO
</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>ループを抜ける</td>
        <td><code>break;</code></td>
        <td></td>
        <td><code>EXIT</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>次のイタレーションへ</td>
        <td><code>continue;</code></td>
        <td></td>
        <td><code>CYCLE</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>入出力のループ</td>
        <td><pre>for (i = 0; i < N; i++) {
  printf("%d\n", a[i]);
}</pre>
        </td>
        <td>
<pre>DO 10 i = 1, N, 1
  PRINT *, a(i)
10 CONTINUE
</pre><br>または<br><code>PRINT *, (a(i), i=1,N,1)</code><br>または<br>
<code>PRINT *, a</code>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

* 入出力並びにDO形並びを使うことができるが、別に使わずに配列名だけ書いてもよい
  * その場合はメモリの並び順にアクセスされる

# 関数
Fortranには他の言語で言う"関数"が2種類ある。
1つは関数でもう1つはサブルーチンである。(2つまとめて手続き副プログラムと呼ぶ)  
両者は似ているが、関数は値を1つ返すのに対し、サブルーチンは明に値を返すわけではない(void型関数みたいなもの)。  
(ちなみに副プログラムには他に初期値設定副プログラムというのがある。)

詳細はCとFortranの違い(言語固有機能編)で触れるとして、以下では関数の基本的な部分に絞って比較する。

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>主プログラム定義</td>
        <td>
<pre>int main(void) {
  /* 処理 */
  return 0;
};</pre>
        </td>
        <td>
<pre>[PROGRAM prog]
! 処理
END [PROGRAM]</pre>
        </td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>関数定義</td>
        <td>
<pre>int add(int a, int b) {
  return a + b;
}</pre>
        </td>
        <td>
<pre>FUNCTION add(a, b)
  INTEGER add, a, b
  add = a + b
  [RETURN]
END [FUNCTION]</pre><br>または<br><pre>INTEGER FUNCTION add(a, b)
  INTEGER a, b
  add = a + b
  [RETURN]
END [FUNCTION]</pre>
        </td>
    </tr>
    <tr>
        <td>プロトタイプ(引用仕様)宣言</td>
        <td><code>int add(int a, int b);</code></td>
        <td><pre>INTEGER add
EXTERNAL add</pre>
        </td>
        <td>(同左)<br>または<br><pre>INTERFACE
  FUNCTION add(a, b)
    INTEGER add, a, b
  END [FUNCTION]
END INTERFACE</pre>
        </td>
        <td>(同左)<br>または<br><code>PROCEDURE(INTEGER) add</code>
        </td>
    </tr>
    <tr>
        <td>関数呼出し</td>
        <td><code>sum = add(sum, elm);</code></td>
        <td><code>sum = add(sum, elm)</code></td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>関数テンプレート<br>(総称名/個別名)</td>
        <td><pre>template &lt;typename T&gt;
T add(T a, T b) {
    return a + b;
}</pre>
        </td>
        <td></td>
        <td><pre>INTERFACE add
  FUNCTION iadd(a, b)
    INTEGER iadd, a, b
  END FUNCTION
  FUNCTION radd(a, b)
    REAL radd, a, b
  END FUNCTION
END INTERFACE
 :
</pre>
        </td>
    </tr>
</table>

* Fortranの場合、`RETURN`文で値を返すのではなく、関数名に戻り値を代入する
  * `RETURN`に整数式を渡すこともできるが、それは戻り値ではない(選択戻り)
* 外部関数を実引数として使うには`EXTERNAL`文が必要
  * ただし引数の型チェックなどがされないため、代わりに`INTERFACE`文を使うことを推奨
  * `PROCEDURE`文は主に手続ポインタに使用されるものだが、`EXTERNAL`文と同じ感覚で使うこともできる  
  (というか`PROCEDURE`文がどういうものかいまいち掴みかねてる)
* `INTERFACE`文はこのほかに演算子のオーバーロードにも使われる
* F77では副プログラム名は副プログラム内部で使えないので再帰関数が書けなかった
  * F90以降もこのルール自体は残っているが、`RECURSIVE`をつければ書けるようになっている
* C++により近い関数テンプレート機能はF202Yの目玉機能の一つとして検討されている

## 文関数
手続きと呼ばれるもの以外にユーザ関数を定義する方法がある。
それが文関数である。言ってしまえばC++のラムダ式のようなもの。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| 関数オブジェクトの生成 | `auto func = [](auto x){ return x + 1; };` | `func(x) = x + 1` | (同左) | | | |

* ただし文関数は廃止予定事項である
  * 代わりに内部副プログラムというものが導入され、推奨されている

# 分割コンパイル
## 共通モジュール
CだとヘッダーファイルだがFortranではモジュールと呼ばれる。
が、機能的にはFortranのモジュールはPythonのモジュールに近いので、この節に限ってはC/C++ではなくPythonと比較する。

<table>
    <tr>
        <th>機能</th>
        <th>Python</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>インポート<br>(モジュール引用)</td>
        <td><code>import mod</code>
        </td>
        <td></td>
        <td><code>USE mod</code>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>一部読込み<br>(参照限定)</td>
        <td><code>from mod import hoge as h</code>
        </td>
        <td></td>
        <td><code>USE mod, ONLY: h=>hoge</code>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>モジュールの宣言</td>
        <td>特になし<br>(ファイルがモジュールの単位)
        </td>
        <td></td>
        <td>
<pre>MODULE mod
  INTEGER hoge
  CONTAINS
  SUBROUTINE foo
  END SUBROUTINE
END MODULE
</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

* モジュール内の引用仕様でモジュール内の手続を使う場合は`MODULE PROCEDURE`文を使う
* F08でサブモジュールというのが追加された
  * 引用仕様と実装の中身を分けて書ける
  * `MODULE`接頭辞という概念が出てきて`MODULE PROCEDURE`文の意味が変わった？

### 文字列の取込み(インクルード)
上で書いたことは半分くらい嘘で、実はC/C++と同じようなインクルードの機能もある。

元は拡張機能だったがF90で標準規格になったものであるが、あまり推奨されていない感じはある。
実際、指定されたファイルの中身をそのまま展開するだけなので、機能がしょぼいどころか場合によっては整合性が取れなくなってコンパイルできなくなることもあり得るため使わない方が無難だろう。

| 機能 | C/C++ | F77 | F90/95 | F03 | F08 | F18 |
| --- | --- | --- | --- | --- | --- | --- |
| インクルード | `#include "header.h"` | | `INCLUDE 'hoge.inc'` | (同左) | (同左) | (同左) |

ちなみに標準規格には規定がないがプリプロセッサ指令(`#include`)を使うこともできそう。
できることは大体同じ。

# その他
## 文の書き方

<table>
    <tr>
        <th>機能</th>
        <th>C/C++</th>
        <th>F77</th>
        <th>F90/95</th>
        <th>F03</th>
        <th>F08</th>
        <th>F18</th>
    </tr>
    <tr>
        <td>行末までコメント</td>
        <td><code>// This is comment.</code></td>
        <td><code>C</code>または<code>*</code><br>(いずれも固定形式の行頭のみに置ける)</td>
        <td>(同左)<br>または<br><code>! This is comment.</code></td>
        <td><code>! This is comment.</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>文の継続</td>
        <td>
<pre>int \
x;</pre>
        </td>
        <td>
<pre>      INTEGER
     + x</pre>
        </td>
        <td>(同左)<br>または<br>
<pre>INTEGER &
x</pre>
        </td>
        <td>
<pre>INTEGER &
x</pre>
        </td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
    <tr>
        <td>文の区切り</td>
        <td><code>int x; float y;</code></td>
        <td></td>
        <td><code>INTEGER x; REAL y</code></td>
        <td>(同左)</td>
        <td>(同左)</td>
        <td>(同左)</td>
    </tr>
</table>

* 固定形式の継続行は、6桁目に空白でも`0`でもない文字が書かれていれば前の行からの継続である

<style type="text/css">
table {
  display: block;
  overflow-x: scroll;
  white-space: nowrap;
}
</style>
