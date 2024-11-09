#!/usr/bin/env bash
SHDIR=$(cd $(dirname $BASH_SOURCE); pwd)
cd $SHDIR/..
bundle exec jekyll serve --drafts
