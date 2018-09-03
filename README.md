# Transformer

An implementation of [Transformer](https://arxiv.org/abs/1706.03762) use Pytorch , and use [Torchtext](https://github.com/pytorch/text) to preprocess IWSLT16 datasets , this project is for learning objectives and also fixed mask issue in [Kyubyong/transformer](https://github.com/Kyubyong/transformer).

- Torchtext.ipynb : make dataset and dataloader
- model.py : Define Transformer
- Testing.ipynb : load [pretrain weights](https://drive.google.com/open?id=1mAo4K-z-_X70R7XjjJ012DoRiYkZxNft) to test 

Testing result example : 

```
Target : <SOS> it 's the symbol of all that we are and all that we can be as an astonishingly <EOS>
Inference : <SOS> it 's the symbol of all of what we are and what we 're an amazing network for <EOS>

Target : <SOS> and of course , we all share the same adaptive imperatives . <EOS> 
Inference : <SOS> and of course , we share the same educational systems . <EOS>

Target : <SOS> we 're all born . we all bring our children into the world . <EOS>
Inference : <SOS> we 're all born in the world . we 're bring kids into the world . <EOS>

Target : <SOS> but what 's interesting is the unique cadence of the song , the rhythm of the dance in <EOS>
Inference : <SOS> but interestingly , the unique cadence of the song , of the rhythm of dance in each one <EOS>
```

## Reference

- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
- [Kyubyong/transformer](https://github.com/Kyubyong/transformer)

