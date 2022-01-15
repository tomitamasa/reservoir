# 概要

某所の課題のために作成した簡素なリザバーコンピューティング（RC）のコードです．
エノン写像のシステム同定（→再帰的予測）ができることを確認しています．

動作確認環境(MacbookAir M1 2020)
- Apple clang version 13.0.0 (clang-1300.0.29.30)
- Target: arm64-apple-darwin21.2.0


# 注意点

最低限，エノン写像のシステム同定ができることを確認しました（失敗することもある）が，
アルゴリズム及びコードの品質について保証はしません．

- ニューロン間の結合は,Erdős–Rényi model(ERモデル)を使いたかったのですが，かなり適当になってしまってます．

- 乱数は，下記のサイトで公開しているrandom.hを使っているので，ダウンロードして同階層に保存する必要があります．

  C言語による乱数生成: https://omitakahiro.github.io/random/random_variables_generation.html

逆行列を掃き出し法で求めています（たまに0の行が出てきて計算不能になるので困ったものです）．

入力重みの構成など，異なるところはありますが以下[1]が参考になると思います．

# 参考

[1]Lukoševičius, M. (2012) ‘A Practical Guide to Applying Echo State Networks’, in Montavon, G., Orr, G.B., and Müller, K.-R. (eds) Neural Networks: Tricks of the Trade: Second Edition. Berlin, Heidelberg: Springer Berlin Heidelberg, pp. 659–686.
