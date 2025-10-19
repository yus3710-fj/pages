---
layout: post
title:  "Fortranにおける関数ポインタ"
date:   2025-10-19 12:37:12 +0900
categories: プログラミング言語
tag: Fortran
---
## 導入
Fortranの関数ポインタの理解に苦労したので、使い方をまとめたい。
インターネットにある日本語記事だと(自分は)理解できなかったので、同じような方の助けになればと思い、筆を執った。

この記事では、関数ポインタの使い道ではなく、関数ポインタを使いたいときに**どう書けばいいか**に焦点を当てる。
(おそらく前者はいろいろ情報が転がっているだろう。)
別の言い方をすると、何か実現したい機能があってそれを関数ポインタを使って解決するという話ではなく、単に関数ポインタの記法の解説をする。

## 関数ポインタとは
はじめに、関数ポインタについて簡単に説明しておく。

関数ポインタとはその名の通り、関数のエイリアスである。
これによって呼び出す関数を動的に切り替えることができるようになる。
(単純な場合はif文で切り替えてもいいが、そうはいかない場合もある。)
以降で示すコード例はいずれも実用性皆無であることに注意されたい。

## 仮手続
一番シンプルなやり方は、手続の引数に手続を渡す方法である。
字面上はポインタの概念が現れないが、実態はれっきとしたポインタである。

```fortran
subroutine sub(func_ext)
  implicit none
  integer :: func_ext
  print *, func_ext()
end subroutine sub

function func()
  implicit none
  integer :: func
  func = 1
end function func

program main
  implicit none
  integer, external :: func
  call sub(func)
end program
```

実引数として手続を渡す場合、EXTERNAL属性が必要である。

### EXTERNAL文と引用仕様
上記の例ではEXTERNAL属性を明示的に指定することで実引数に指定できるようにしたが、これはFORTRAN 77の頃の古い記法である。
Fortran 90以降では代わりに引用仕様(INTERFACE文)を使うことが推奨される。
(今回の例のように引数が単純なら使わなくてもよいが、それ以外の場合は引用仕様を書くことが**必須**である。書かないとバグる。)

引用仕様を使って書き換えた例を以下に示す。

```fortran
subroutine sub(func_ext)
  implicit none
  interface
    function func_ext()
      implicit none
      integer :: func_ext
    end function func_ext
  end interface
  print *, func_ext()
end subroutine sub

function func()
  implicit none
  integer :: func
  func = 1
end function func

program main
  implicit none
  interface
    function func()
      implicit none
      integer :: func
    end function func
  end interface
  call sub(func)
end program
```

これを見て怠惰なプログラマ諸君はこう思ったに違いない。
「同じような引用仕様を何度も書かされるのは御免だ」と。
そんな諸君のためにFortran 2003では抽象引用仕様というのが追加されている。
要は関数の型エイリアスである。

ついでにモジュールも使って以下のように書き換えた。

```fortran
module mod
  abstract interface
    function noarg_retint()
      integer :: noarg_retint
    end function noarg_retint
  end interface

  contains
  subroutine sub(func_ext)
    implicit none
    procedure(noarg_retint) :: func_ext
    print *, func_ext()
  end subroutine sub
end module mod

function func()
  implicit none
  integer :: func
  func = 1
end function func

program main
  use mod
  implicit none
  procedure(noarg_retint) :: func ! EXTERNAL属性は勝手に付与される
  call sub(func)
end program
```

ABSTRACT INTERFACE文で抽象引用仕様を宣言し、手続宣言文(PROCEDURE文)にその引用仕様名を指定する。

## 手続ポインタ
これまで見てきたように、呼び出す関数を動的に変える方法はFORTRAN 77から存在はしていた。
が、引数として渡すしか手段がないというのは実用上(特にOOPにおいて)不便なため、Fortran 2003で手続ポインタが導入された。

ただ、特別なことは何もなく、POINTER属性を追加で指定するだけである。
実際、手続ポインタは「EXTERNAL属性及びPOINTER属性をもつ手続」と定義されている。

* EXTERNAL文
    ```fortran
    subroutine sub(func_ext)
      implicit none
      integer, external, pointer :: func_ext
      print *, func_ext()
    end subroutine sub
    
    function func()
      implicit none
      integer :: func
      func = 1
    end function func
    
    program main
      implicit none
      integer, external :: func
      integer, external, pointer :: fptr
      interface
        subroutine sub(func_ext)
          implicit none
          integer, external, pointer :: func_ext
        end subroutine sub
      end interface
      fptr => func
      call sub(fptr)
    end program
    ```
    仮引数がPOINTER属性を持つ場合、引用仕様が必須。

* 引用仕様
    ```fortran
    subroutine sub(func_ext)
      implicit none
      interface
        function func_ext()
          implicit none
          integer :: func_ext
        end function func_ext
      end interface
      pointer :: func_ext
      print *, func_ext()
    end subroutine sub
    
    function func()
      implicit none
      integer :: func
      func = 1
    end function func
    
    program main
      implicit none
      interface
        function func()
          implicit none
          integer :: func
        end function func
        function fptr()
          implicit none
          integer :: fptr
        end function fptr
        subroutine sub(func_ext)
          implicit none
          pointer :: func_ext ! gfortranはなぜかこちらに書かないとエラー
          interface
            function func_ext()
              implicit none
              integer :: func_ext
            end function func_ext
          end interface
          !pointer :: func_ext
        end subroutine sub
      end interface
      pointer :: fptr
      fptr => func
      call sub(fptr)
    end program
    ```

* 手続宣言文
    ```fortran
    module mod
      abstract interface
        function noarg_retint()
          integer :: noarg_retint
        end function noarg_retint
      end interface
    
      contains
      subroutine sub(func_ext)
        implicit none
        procedure(noarg_retint), pointer :: func_ext
        print *, func_ext()
      end subroutine sub
    end module mod
    
    function func()
      implicit none
      integer :: func
      func = 1
    end function func
    
    program main
      use mod
      implicit none
      procedure(noarg_retint) :: func
      procedure(noarg_retint), pointer :: fptr
      fptr => func
      call sub(fptr)
    end program
    ```
    こちらはモジュールから引用仕様を取ってこれるので再度書く必要はない。

※ちなみにOOPに触れるのであれば、型束縛手続も扱う必要があるが、今回は説明しない。
(OOP固有の事情があり、それがまあまあややこしいため。多分OOPの文脈で説明した方がいい。)

## まとめ
呼び出す関数を動的に切り替える方法は、EXTERNAL属性を持つ手続を実引数に指定するほか、Fortran 2003から導入された手続ポインタを使う方法がある。
後者は、EXTERNAL属性を持つ手続にPOINTER属性を追加することで実現できる。
なお、EXTERNAL属性はFortran 2003以降、手続宣言文によっても付与することができる。

### あとがき
最後に、自分は理解する上で何に躓いたかというのを記しておく。

"Fortran 手続ポインタ"で調べると、真っ先にPROCEDURE文の説明が出てくる。
これを見て手続宣言文が手続ポインタのための言語機能だと誤解してしまい、文法的に間違ったコードをたくさん書いてよくコンパイラに怒られていた。
もちろんネットにある例をそのままコピペできる場合はそれで乗り切れるが、仕事柄それ以外の書き方もできるようになる必要があり、苦労した。

自分は運よく手続宣言文と手続ポインタが密接に関係していないことを知るきっかけを得て、そこから手続ポインタ周りの仕様を理解できたが、それがなければ今も誤解したままになっていたと思うので、今回そこをきちんと分けて説明するようにした。
