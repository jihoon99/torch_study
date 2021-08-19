import os
import torchtext
version = list(map(int, torchtext.__version__.split('.')))
if version[0] <= 0 and version[1] < 9:
    from torchtext import data, datasets
else:
    from torchtext.legacy import data, datasets

PAD, BOS, EOS = 1, 2, 3
# PAD의 번호는 1, BOS는 2, EOS는 3인가보네


# DataLoader : Field {src, tgt} -> TranslationDataset {train, valid} -> BucketIteration {train_iter, valid_iter}
# nex(iter(train_iter)) -> 
'''
(tensor([[ 292,   11, 1603,  ...,  117,  140,    4],
        [  23,   97,    5,  ...,    1,    1,    1],
        [  23, 1373,   78,  ...,    1,    1,    1],
        ...,
        [  42,   18,  318,  ...,    1,    1,    1],
        [1171, 1236,  346,  ...,    1,    1,    1],
        [ 304,  203,    8,  ...,    1,    1,    1]]),
tensor([90, 86, 83, 67, 66, 66, 64, 61, 59, 58, 57, 54, 54, 53, 53, 53, 53, 51,
        51, 51, 51, 50, 49, 49, 49, 48, 47, 47, 46, 45, 44, 44, 42, 42, 41, 41,
        40, 39, 39, 37, 37, 36, 36, 36, 34, 33, 33, 33, 32, 32, 31, 31, 31, 30,
        29, 29, 28, 27, 26, 25, 25, 24, 23, 23, 22, 21, 21, 21, 20, 20, 20, 20,
        19, 18, 18, 17, 17, 16, 16, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12,
        12, 12, 11, 11, 11,  9,  9,  8,  6,  6]))
두번째 텐서가 뭔지 모르겟네.     
        '''

class DataLoader():

    def __init__(self,
                 train_fn=None,
                 valid_fn=None,
                 exts=None,
                 batch_size=64,
                 device='cpu',
                 max_vocab=99999999,
                 max_length=255,
                 fix_length=None,
                 use_bos=True,
                 use_eos=True,
                 shuffle=True,
                 dsl=False
                 ):

        super(DataLoader, self).__init__() # ??? 상속받을게 없는데?

        # Field -> fields -> TabularDataset -> build_vocab -> Bucket

        # src와 tgt가 각각 있는 이유는, 파일이 각각 있었기 때문이다.
            # torchtext.data.Field
        self.src = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length, # None
            init_token='<BOS>' if dsl else None, # dsl : dure learning할때 필요한것. 지금은 None이라고 보면 됨.
            eos_token='<EOS>' if dsl else None,
        )

        self.tgt = data.Field(
            sequential=True,
            use_vocab=True,
            batch_first=True,
            include_lengths=True,
            fix_length=fix_length,
            init_token='<BOS>' if use_bos else None, # True .. learning에서는 필요 없고, 생성자 할때만 필요함(?)
            eos_token='<EOS>' if use_eos else None,
        )

        if train_fn is not None and valid_fn is not None and exts is not None:
            # TranslationDataset는 밑에 정의 되어있습니다.
            train = TranslationDataset(
                path=train_fn, # train file path
                exts=exts, # en,ko path가 튜플로 들어가 있음.
                fields=[('src', self.src), ('tgt', self.tgt)], # 사용할 필드 명
                max_length=max_length
            )
            valid = TranslationDataset(
                path=valid_fn,
                exts=exts,
                fields=[('src', self.src), ('tgt', self.tgt)],
                max_length=max_length,
            )

            # bucketIterator가 하는 일을 실제 데이터를 가지고 와서. -> pad까지 체운 형태로 만들고
            # 미니배치 단위로 만들어주는 역할을 한다.
            # https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator
            self.train_iter = data.BucketIterator(
                train,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=shuffle,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)), # ?????????????? what's x?
                sort_within_batch=True,
            )
            # next(iter(train_iter)) 하면, 

            self.valid_iter = data.BucketIterator(
                valid,
                batch_size=batch_size,
                device='cuda:%d' % device if device >= 0 else 'cpu',
                shuffle=False,
                sort_key=lambda x: len(x.tgt) + (max_length * len(x.src)),
                sort_within_batch=True,
            )

            self.src.build_vocab(train, max_size=max_vocab)
                # construct the vocab object for this field from one or more datasets.
                # https://torchtext.readthedocs.io/en/latest/data.html
                # it's word2idx : 어떤 단어가 몇번째 인덱스로 맵핑되는지.
            self.tgt.build_vocab(train, max_size=max_vocab)

    def load_vocab(self, src_vocab, tgt_vocab):
        '''?????????????????????????????????????'''
        self.src.vocab = src_vocab
        self.tgt.vocab = tgt_vocab


# torchtext에는 maxlen을 잘라주는게 없어서 customizing했어.
class TranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, fields, max_length=None, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            # fields가 [('src',src),('tgt',tgt)]형태가 아닐때 다시 정의를 함.
            fields = [('src', fields[0]), ('trg', fields[1])]

        if not path.endswith('.'):
            # 주소의 끝에 .이 없다면 추가로 넣어줘.
            path += '.'

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)

        examples = []
        with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
            # src, trg path에서 파일을 불러오고 한줄씩 for문
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip() # 오른쪽끝 스페이스 제거.
                # max_length가 있을 경우에는 작업을 해줌.
                if max_length and max_length < max(len(src_line.split()), len(trg_line.split())): 
                    # 스페이스를 띄어쓰기라고 가정, max_len보다 클때(?) 이부분 잘못된거 같은데...
                    '''?????????????????????????????????????????'''
                    continue
                if src_line != '' and trg_line != '':
                    # 별일 없을때 examples에 데이터를 추가.
                    examples += [data.Example.fromlist([src_line, trg_line], fields)]

        super().__init__(examples, fields, **kwargs)


if __name__ == '__main__':
    import sys
    loader = DataLoader(
        sys.argv[1],
        sys.argv[2],
        (sys.argv[3], sys.argv[4]),
        batch_size=128
    )

    print(len(loader.src.vocab))
    print(len(loader.tgt.vocab))

    for batch_index, batch in enumerate(loader.train_iter):
        print(batch.src)
        print(batch.tgt)

        if batch_index > 1:
            break
        '''???????????????????????????????????????????????????????'''
        # batch.src의 shape이 계속 바뀌는데... 어떻게 처리할까?

