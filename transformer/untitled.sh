
'''
리눅스 명령어

wc : 사용자가 지정한 파일의 행, 단어, 문자수를 세는 프로그램
wc -l file_name
에) wc -l ./*.txt

ls -l : 권한 소유자 갱신일 확인
<-> ll
'''
#————————————————————————————
'''
head filename : 
예) head 1_구어체.txt
'''

'''
1, T-V-T 나누기
cat *.txt > corpus.tsv
head -n 5 corpus.tsv     파일 순서대로 concat 되있을것이란 말이지
shuf corpus.tsv > corpus.shuf.tsv
head -n 5 ./corpus.shuf.tsv

head -n 1200000 corpus.shuf.tsv > corpus.shuf.train.tsv
tail -n 402409 corpus.shuf.tsv | head -n 200000 > corpus.shuf.valid.tsv  # 뒤에 402409로 df를 하나 만들고, 그중 20000개를 Valid
tail -n 202409 corpus.shuf.tsv > corpus.shuf.test.tsv
'''

————————————————————————————
2. 전처리

cut [옵션] [파일명]

cut -f1 corpus.shuf.train.tsv : 첫번째 필드만 가져와.


cut -f1 corpus.shuf.train.tsv > corpus.shuf.train.ko ; cut -f2 corpus.shuf.train.tsv > corpus.shuf.train.en

——————————————————————————————

3. 토크나이저

예) head -n 5 corpus.shuf.train.ko | mecab -O wakati

-test ko에 대해 토크나이즈
cat ./data/corpus.shuf.test.ko | mecab -O wakati -b 99999 | python ./nlp_preprocessing/post_tokenize.py ./data/corpus.shuf.test.ko > ./data/corpus.shuf.test.tok.ko


-영어에 대해서도 해줌. _있는 부분이 원래 띄어쓰기가 있는 부분임.
cat ./data/corpus.shuf.test.en | python ./nlp_preprocessing/tokenizer.py | python ./nlp_preprocessing/post_tokenize.py ./data/corpus.shuf.test.en > ./data/corpus.shuf.test.tok.en &

- 트레인 코리안
cat ./data/corpus.shuf.train.ko| mecab -O wakati | python ./nlp_preprocessing/post_tokenize.py ./data/corpus.shuf.train.ko > ./data/corpus.shuf.train.tok.ko


cat ./data/corpus.shuf.train.en | python ./nlp_preprocessing/tokenizer.py | python ./nlp_preprocessing/post_tokenize.py ./data/corpus.shuf.train.en > ./data/corpus.shuf.train.tok.en &


cat ./data/corpus.shuf.valid.ko | mecab -O wakati -b 99999 | python ./nlp_preprocessing/post_tokenize.py ./data/corpus.shuf.valid.ko > ./data/corpus.shuf.valid.tok.ko &


——————————————————————————————————

4. Data Preparation
1) git clone https://github.com/kh-kim/subword-nmt
	-> rsennrich 교수님거 조금 수정한 버전 -> 이걸 하면 _ 생성.
	-> 이미 우리건 ‘_’가 들어가 있음.
	-> learn_bpe, apply_bpe를 많이 사용할 것.

2). 
python ./subword-nmt/learn_bpe.py --asdf : 어떤 것을 할 수 있는지 옵션들이 나옴.

python ./subword-nmt/learn_bpe.py --input ./data/corpus.shuf.train.tok.en --output bpe.en.model --symbols 50000 --verbose    # output을 bpe.en.model로 저장하고, 50000번의 merge operation을 수행한다. verbose를 켜서 수행하는 모습을 보여준다.
영어 같은경우 50000만을 하면 vocab size가 2~3만개로 된다.

3).결과물 살펴보자
head -n 5 bpe.en.model  # merge instruction이 쓰여있다.
wc -l ./bpe.en.model


4)
한국어도 bpe적용하자
python ./subword-nmt/learn_bpe.py --input ./data/corpus.shuf.train.tok.ko --output bpe.ko.model --symbols 30000 --verbose # 한국어는 영어에 비해서 좀더 복잡하기 때문에 30000번으로 충분함.
# 너무 쓸모없는 것 까지 합쳐지면, 저 숫자를 낮춰야함.


5)
python ./subword-nmt/apply_bpe.py --asdf # 옵션 보기

head -n 5 ./data/corpus.shuf.train.tok.ko | python subword-nmt/apply_bpe.py -c ./bpe.ko.model # bpe 적용하기..
▁▁줄 리아 ▁가 ▁▁그녀 ▁의 ▁▁컴퓨터 ▁로 ▁▁엘라 ▁의 ▁▁웹 ▁사이트 ▁에 ▁▁들어 갔 ▁어 ▁.
▁▁김용 균 ▁▁씨 ▁의 ▁▁사망 ▁사고 ▁가 ▁▁발생 ▁한 ▁▁지 ▁▁6 ▁일 ▁▁만 ▁이 ▁다 ▁.
▁▁특별 ▁한 ▁▁사람 ▁은 ▁▁있 ▁지만 ▁▁여자 ▁▁친구 ▁는 ▁▁아니 ▁야 ▁.
▁▁온라인 ▁▁사 교육 ▁시장 ▁이 ▁▁급 속도 ▁로 ▁▁확대 ▁되 ▁고 ▁▁있 ▁다며 ▁▁종목 ▁▁매수 ▁를 ▁▁추천 ▁하 ▁는 ▁▁내용 ▁이 ▁었 ▁다 ▁.
▁▁이 ▁밖 ▁에 ▁도 ▁▁도 ▁는 ▁▁인 명 ▁피해 ▁가 ▁▁우려 ▁되 ▁는 ▁▁지역 ▁에 ▁▁대한 ▁▁사전 ▁▁예 찰 ▁▁활동 ▁을 ▁▁실시 ▁하 ▁는 ▁▁한 편 ▁, ▁▁집 중 호우 ▁시 ▁▁신속 ▁한 ▁▁대 피 ▁를 ▁▁위해 ▁▁설치 ▁한 ▁▁재난 ▁▁예 ▁경보 ▁▁시스템 ▁의 ▁▁상시 ▁▁가동 ▁태세 ▁를 ▁▁유지 ▁하 ▁고 ▁▁있 ▁다 ▁.

#tokenizer하면 _ 하나만 들어가고, subword까지 먹으면 두개가 들어감. -> __ 두개가 있으면 띄어쓰기이다.
# __유 독 -> 이런걸 합치고 싶으면, 30000에서 더 늘려..


6) 모두다에 적용
cat ./data/corpus.shuf.train.tok.ko | python subword-nmt/apply_bpe.py -c bpe.ko.model > ./data/corpus.shuf.train.tok.bpe.ko ; cat ./data/corpus.shuf.valid.tok.ko | python subword-nmt/apply_bpe.py -c bpe.ko.model > ./data/corpus.shuf.valid.tok.bpe.ko ; cat ./data/corpus.shuf.test.tok.ko | python subword-nmt/apply_bpe.py -c bpe.ko.model > ./data/corpus.shuf.test.tok.bpe.ko

cat ./data/corpus.shuf.train.tok.en | python subword-nmt/apply_bpe.py -c bpe.en.model > ./data/corpus.shuf.train.tok.bpe.en ; cat ./data/corpus.shuf.valid.tok.en | python subword-nmt/apply_bpe.py -c bpe.en.model > ./data/corpus.shuf.valid.tok.bpe.en ; cat ./data/corpus.shuf.test.tok.en | python subword-nmt/apply_bpe.py -c bpe.en.model > ./data/corpus.shuf.test.tok.bpe.en







1. simple_NMT/models/seq2seq.py
--------------------------------------------------
	여기서 모델 아키텍쳐를 만들고


2. simple_NMT/trainer.py
--------------------------------------------------
	여기서 이그나이트 함.(?)
	https://pytorch.org/ignite/

3. simple_NMT/data_loader.py
--------------------------------------------------
	DataLoader 만들기

4. train.py
--------------------------------------------------
	wrapping

5. cmd : terminal
--------------------------------------------------
watch -n .5 nvidia-smi        :  nvidia cudnn있어야함.
top                           : 현 리소스 사용량

mkdir models/models.20200906

python train.py --asdf

python train.py --train ./data/corpus.shuf.train.tok.bpe --valid ./data/corpus.shuf.valid.tok.bpe --lang enko --gpu_id 0 --batch_size 160 --n_epochs 30 --max_length 64 --dropout .2 --word_vec_size 512 --hidden_size 768 --n_layer 4 --max_grad_norm 1e+8 --iteration_per_update 2 --lr 1e-3 --lr_step 0 --use_adam --model_fn ./models/models.20200906/enko.bs-160.max_length-64.dropout-2.ws-512.hs-768.n_layers-4.iter_per_update-2.pth


6. continue_training.py
--------------------------------------------------
- trainer.py에 torch.save를 보면 model, opt, config, src_vocab, tgt_vocab등을 pt로 세이브함.
- train.py에 define_argparser를 보면 is_continue = False로 되어있다.
	True로 주면 --load_fn이라는게 생김.
	True이면 add_argument에 있는 required가 not True가 되어버림.
- continue_train.py
	if __name__ == '__main__':
		config = define_argparser(is_continue=True) # 이것부터 실행
		# 조심할것은 p.add_argument('--init_epoch'에서 값을 19 막 이렇게 줘야해 when saved model was ended at 18th epochs b/c learning scheduler
		continue_main(config, main) # 위에서 받은 config를 continue_main에 전달함

- continue_train.py 쓰는 방법
cmd : python continue_train.py --load_fn ./models/models.20200906/enko.transformer.bs-128.max_length-64.dropout-2.hs0768.n_layers-4.iter_per_update-16.random.03.1.87-6.48.1.79-5.96.pth --init_epoch 4 --max_grad_norm 1e+7



7. 추론을 하기위한 method를 작성해보자 : seq2seq.py
---------------------------------------------------
- search method만듦


8. 추론을 하기위한 : translate.py
----------------------------------------------------
- beam_search : 1로 해야함.(아직 beam_search안배움.)
- 


9. cmd
----------------------------------------------------
head -n 1 ./corpus.shuf.test.tok.bpe.en : 데이터 뭐있나 확인

- detokenize 하는 법
head -n 1 ./data/corpus.shuf.test.tok.bpe.en | python ./nlp_preprocessing/detokenizer.py

head -n 5 ./data/corpus.shuf.test.tok.bpe.en | python translate.py --model_fn ./models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.-3.67.1.46-4.31.pth --gpu_id -1 --batch_size 2 --beam_size 1 | python ./nlp_preprocessing/detokenizer.py
																					# en -> ko, dropout :.3,  wordsize = 512, hidden 768, iteration update : 2, batch 96 ==> 192번마다 grad update					# batch:2개로 inference
																																										# 30번째 epoch
																																										# train_loss : 1.3
																																										# train perplexity 3.67
																																										# validation_loss : 1.46
																																										# valid perplexity : 4.31
head -n 5000 | tail -n 5 | ./data/corpus.shuf.test.tok.bpe.en | python translate.py --model_fn ./models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.-3.67.1.46-4.31.pth --gpu_id -1 --batch_size 2 --beam_size 1 | python ./nlp_preprocessing/detokenizer.py




10. BLEU 구하기 : cmd
---------------------------------------------------
정석 : 모든 epoch에 대해 validation의 BLEU를 구하고, 가장 좋은 친구를 test set에 적용함.

- 하지만 200,000문장이라서 BLEU구하는게 힘듦. -> 대략 1,000문장 정도만 해보자.
	head -n 1000 corpus.shuf.test.tok.bpe.en > ./corpus.shuf.test.tok.bpe.head-1000.en
	head -n 1000 corpus.shuf.test.tok.bpe.ko > ./corpus.shuf.test.tok.bpe.head-1000.ko
	head -n 1000 corpus.shuf.valid.tok.bpe.en > ./corpus.shuf.valid.tok.bpe.head-1000.en
	head -n 1000 corpus.shuf.valid.tok.bpe.ko > ./corpus.shuf.valid.tok.bpe.head-1000.ko

	wc -l ./corpus.shuf.*
	head -n 1 ./*.head*

- bash

	MODEL_FN = $1
	GPU_ID = -1
	BEAM_SIZE = 1
	TEST_FN = ./corpus.shuf.valid.tok.bpe.head-1000.en
	REF_FN = ./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko

	cat ${TEST_FN} | python ../translate.py --model ${MODEL_FN} --gpu_id ${GPU_ID} --lang enko --beam_size ${BEAM_SIZE} | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ${REF_FN}
																																									# 여기 파트가 중요하다. // BLEU들어가기전에 토크나이즈 한번 더 해줘야함.
																																														# 파이프를 먹여서 multi-bleu에 들어가는데, // 동시에 정답도 알려줘야함.

- 위 bash에 들어가는 REF_FN을 만들어 주자.
	cat ./corpus.shuf.valid.tok.bpe.head-10000.ko | python ../nlp_preprocessing/detokenizer.py | head -n 2 # 결과물 두개만 확인해 보겠다.
	cat ./corpus.shuf.valid.tok.bpe.head-10000.ko | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati > ./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko
	tail -n 5 ./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko # 확인하게 한다.


	cat ./corpus.shuf.valid.tok.bpe.head-1000.en | python ../nlp_preprocessing/detokenize.py | python ../nlp_preprocessing/tokenizer.py > ./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.en
																									# 영어도 토크나이즈 한번한다.
	cat ./corpus.shuf.test.tok.bpe.head-1000.ko | python ../nlp_preprocessing/detokenize.py | mecab -O wakati > ./corpus.shuf.test.tok.bpe.head-1000.detok.tok.ko
	cat ./corpus.shuf.test.tok.bpe.head-1000.en | python ../nlp_preprocessing/detokenize.py | python ../nlp_preprocessing/tokenizer.py > ./corpus.shuf.test.tok.bpe.head-1000.detok.tok.en


- 잘 햇는지 확인
	head -n 1 ./corpus.shuf.*.detok.tok.*

- translate.py 한번 실행
	head -n 5 ./corpus.shuf.valid.tok.bpe.head-1000.en | python ../translate.py --model_fn ../models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.3.67.1.46-4.31.pth --gpu_id -1 --lang enko --beam_size 1

- bleu 구하기 위한 전처리(?)
	head -n 5 ./corpus.shuf.valid.tok.bpe.head-1000.en | python ../translate.py --model_fn ../models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.3.67.1.46-4.31.pth --gpu_id -1 --lang enko --beam_size 1 | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati

- blue 구하기
	time cat ./corpus.shuf.valid.tok.bpe.head-1000.en | python ../translate.py --model_fn ../models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.3.67.1.46-4.31.pth --gpu_id -1 --lang enko --beam_size 1 | python ../nlp_preprocessing/detokenizer.py | mecab -O wakati | ./multi-bleu.perl ./corpus.shuf.valid.tok.bpe.head-1000.detok.tok.ko


- bash로 BLEU구하가
time ./enko_valid.sh ../models/enko.dropout-3.ws-512.hs-768.iteration_per_update-2.max_length-100.batch_size-96.30.1.30.3.67.1.46-4.31.pth


- massive_test.py로 모든 모델에 대해서 BLEU구하기
python massive_test.py --model_fn ../models/models.20200906/enko.transformer.bs-128.max_length-64.dropout-2.hs-768.n_layers-4.iter_per_update-16.random.* --script_fn ./enko_valid.sh



11. BEAM SEARCH :
----------------------------------------------
다른 강의에서는 Transformer등을 가르치지만, BEAM SEARCH는 deploy하기전에 반드시 해야하는 작업중 하나이다.

seq2seq.py -> beam_search 와 search.py를 살펴보겠다.

1. 인코더 통과  ->  2. BeamSearch초기화 ->  4. Board로 부터 TmpMiniBatch생성 
 (seq2seq.py)      seq2seq				 /		seq2seq.py의 while문   \
										/                              \
						3. 다음 Temp TmpMiniBatch준비					5. 
							search.py and s2s.py