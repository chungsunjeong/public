안녕하세요. 본 임시 github는 "Bi-directional Attention Flow for Machine Comprehension (Seo et al., 2017)" 에 대한 구현 과제를 포함하고 있습니다.
___
## 1. command 및 실행환경
* command
	'python main.py'
* 실행환경
	* python 3.6.4
	* tensorflow 1.13.1
	* nltk 3.2.1
___
## 2. 파일 설명
본 github 내 파일은 크게 다음의 종류로 나뉩니다. (폴더, jupyter notebook 파일, python 파일)
* 폴더
	* 정리를 위한 공간입니다.
	* "data" 폴더: 
		* 구현에 사용되는 데이터셋들을 저장한 공간입니다. 
		* "glove": word to vector 데이터셋인 glove dataset
		* "nltk-data": tokenize 를 하기 위한 dataset
		* "squad": 모델의 training/test 에 사용되는 golden standard dataset. 
			* 기존 파일은 v1.1 이며, 빠르게 테스트를 진행하기 위해 small 셋을 자체적으로 만들었습니다. 
			* 기존 파일 명은 "train-v1.1.json" 과 "dev-v1.1.json" 입니다. 
	* "saved_model" 폴더:
		* squad dataset (train-v1.1.json) 으로 학습이 완료된 모델을 저장한 공간입니다.
		* 최종적으로 학습이 완료된 모델에 대한 meta data: "my_test_model-{2019-07-18-14-23}.ckpt.meta"
		* 최종적으로 학습이 완료된 모델에 대한 full data: "my_test_model-{2019-07-18-14-23}.ckpt"
* jupyter notebook 파일
	* 테스트 실행을 위한 notebook 파일입니다.
	* 임시적으로 코드를 실행하고 연습하는 공간이기 때문에 정리가 잘 안되어 있습니다.
	* "pad_distributed_code (contain copies).ipynb"
		* 이는 이미 구현이 완료된 BiDAF 모델에 대한 원본 코드 일부를 포함하고 있습니다. 
		* 오직 BiDAF 모델에 대한 이해와 참고를 위한 용도입니다.
* python 파일
	* 제가 구현한 python 코드들 입니다.
	* 아래 순서로 보시는 걸 추천드립니다. 
	* 코드 내 일부 함수들만 설명드리겠습니다. 
	1. "load_dataset.py":
		* squad dataset 을 preprocessing 하고 loading 하기 위한 코드
		* 'get_corresponding_glove_word2vec(filename,word_list)'
			주어진 단어 list에 대한 glove word2vec 값과 vector 의 dimension 를 출력
		* 'get_word_char_counter(data)'
			주어진 squad data 에 대해 모든 context와 query들의 word에 대한 counter, char에 대한 counter 를 출력
		* 'get_ind_dictionaries(word_list,char_list)'
			주어진 단어 list와 문자 list 에 대해 각각 index 를 붙여주고 이를 mapping 하는 네 종류의 dictionaries 출력
			word index <-> word, character index <-> character
		* 'get_preprocessed_dataset(data,mode='train')'
			주어진 squad data 에 대해 사용하기 쉽게 preprocessing 하여 각종 dictionaries 출력
			1. config: dataset 에 기반하여 default 로 설정되는 값들 포함
			2. sub_info_dataset: dataset 에 대한 유용한 정보들을 포함
			3. index_dataset: word/char index 에 대한 mapping table 포함
			4. word2vec_dataset: word2vec 에 대한 내용 포함
			5. full_dataset: 직접적으로 모델을 학습하는데 필요한 inputs 이 정리된 dictionary
	2. "layer.py":
		* 모델을 구축하는데 필요한 layer 함수를 포함
		* 'CharEmbLayer()': character embedding layer
		* 'WordEmbLayer()': Word Embedding Layer
		* 'DenseLayer()': single layer neural network (activation 함수 포함)
		* 'HighwayLayer()': Highway Network Layer
		* 'ContextualEmbLayer()': Contextual Embedding Layer
		* 'AttentionLayer()': Attention Layer
		* 'TwoLSTMs_ModelingLayer()': Modeling Layer
		* 'OutputLayer()': Output Layer
	3. "model.py":
		* 논문의 BiDAF 모델에 대한 class를 포함하는 코드
		* <class "my_QAmodel">: 모델에 대한 class 
	4. "main.py":
		* BiDAF 모델을 생성하고 학습하는 코드
	5. "evaluate.py":
		* 학습 완료된 BiDAF 모델을 평가하는 코드
		* 여기에 포함된 함수들은 전부 배포판 evaluation 코드들입니다.
		* **__main__ 에서 저장된 모델을 restore 하고 evaluation 하려고 했습니다. 하지만 restore 하면 kernel 이 죽는 현상 등의 에러로 인해 진행하지 못하였습니다.**
___
## 3. 결과
	1. evaluation
		* 위에서 언급한 대로 evaluation 은 진행하지 못하였습니다. 최종적으로 학습이 완료된 시점이 과제 제출 당일이라 에러를 해결하기에 시간이 부족하였습니다. 이점 죄송하다는 말씀 드리고 싶습니다.
	2. training process
		* 부수적으로 training 진행 과정 동안 loss 가 전체적으로 줄어들고 있음을 확인하였습니다.
		* 다만, 제가 가진 컴퓨터의 한계 상 batch_size 가 30까지 밖에 못하였습니다. (원본 code에서 사용하는 default batch_size = 60)
		* 따라서, batch_size를 조금 더 늘리면 training loss 가 더 줄어들 것으로 보입니다. 
		* 이 밖에도 모델 학습에 영향을 주는 다양한 설정값이나, 모델 개선에 도움을 주는 추가적인 overfitting 방지 기술들(l2 norm, dropout 등)이 있지만, 시간이 부족하여 진행하지 못하였습니다. 죄송합니다. 
