# Sem-GAN

<a href="https://arxiv.org/abs/1807.04409">Sem-GAN: Semantically-Consistent Image-to-Image Translation</a>
<img width="170" alt="s" src="https://user-images.githubusercontent.com/64032115/109528235-f1f8de00-7a60-11eb-88f0-86420de17b75.png">


//Preparation
CycleGANであるdomainAとあるdomainBでの変換をするとき，<br>
１．<a href="https://github.com/Daiki-Shiotsuka/FCN_PyTorch"> セグメンテーション用のネットワーク</a>でdomainAの画像でセグメンテーションのネットワークをトレーニングし，<a href="https://github.com/Daiki-Shiotsuka/FCN_PyTorch">そこ</a>のcheckpointsのlatset.pthを<a href="https://github.com/Daiki-Shiotsuka/SemSeg_CycleGAN_PyTorch">本コード</a>のpretrainedのdomainAにいれる．<br>
2. domainBについても同様にする．<br>
3. Train<br>
4. Test<br>

//Trainig<br>
python train.py --dataset_name dataset_name<br>

//Test<br>
python test.py --dataset_name dataset_name<br>

#dataset_name is a dataset name you use.


ただ，今回の実装ではbasicなCycleGANよりクオリティは落ちたように感じました．<br>
basicなCycleGANは<a href="https://github.com/Daiki-Shiotsuka/CycleGAN_PyTorch">ここ</a>から試せます。
