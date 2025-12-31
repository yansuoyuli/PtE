import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import random
import time

class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'TRAIN ITEM', args.item_trn, 'TEST ITEM', args.item_tst)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		log('Model Initialized')

		recallMax_20 = 0
		ndcgMax_20 = 0
		recallMax_40 = 0
		ndcgMax_40 = 0
		bestEpoch = 0

		time_counter = 0

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			
			# timer begin
			start_time = time.time()
			reses = self.trainEpoch(ep)
			end_time = time.time()
			running_time = end_time - start_time
			time_counter += running_time
			# timer end

			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if (reses['Recall@20'] > recallMax_20):
					recallMax_20 = reses['Recall@20']
					ndcgMax_20 = reses['NDCG@20']
					recallMax_40 = reses['Recall@40']
					ndcgMax_40 = reses['NDCG@40']
					bestEpoch = ep
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		print('Best epoch : ', bestEpoch, ' , Recall@20 : ', recallMax_20, ' , NDCG@20 : ', ndcgMax_20, ', Recall@40 : ', recallMax_40, ', NDCG@40 : ', ndcgMax_40)
		print('Average Training Time: ', time_counter/args.epoch)

	def saveUserEmbedding(self):
		with torch.no_grad():
			usrEmbeds, _ = self.model.forward_cl(self.handler.torchBiAdj_trn)
		usrEmbeds = usrEmbeds.cpu().numpy()
		f = open('./../encoder/user_embedding/netflix/userEmbeds.pkl', 'wb')
		pickle.dump(usrEmbeds, f)
		f.close()
		print("save best usr Embedding")

	def prepareModel(self):
		self.model = Model().cuda()
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		self.model.setItemFeatsTrn(self.handler.item_feats_trn.detach())
		self.model.setItemFeatsTst(self.handler.item_feats_tst.detach())
		if args.user_aug == 1:
			self.model.setItemFeatsTrnProfile(self.handler.item_feats_trn_profile.detach())
			self.model.setItemFeatsTstProfile(self.handler.item_feats_tst_profile.detach())
		self.model.setUserFeats(self.handler.user_feats.detach())
	
	def trainEpoch(self, ep):
		self.model.train()

		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		epLoss, epRecLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch
		for i, tem in enumerate(trnLoader):
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()
			poss = poss.long().cuda()
			negs = negs.long().cuda()
			usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj_trn)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
			
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
			regLoss =(calcRegLoss(self.model)) * args.reg

			loss = bprLoss + regLoss
			epLoss += loss.item()
			epRecLoss += bprLoss.item()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()
			log('Step %d/%d: loss = %.3f, regLoss = %.3f         ' % (i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epLoss / steps
		ret['recLoss'] = epRecLoss / steps
		return ret

	def testEpoch(self):
		self.model.eval()

		tstLoader = self.handler.tstLoader
		epRecall_20, epNdcg_20, epRecall_40, epNdcg_40 = [0] * 4
		i = 0
		num = tstLoader.dataset.__len__()
		steps = num // args.tstBat

		with torch.no_grad():
			usrEmbeds, _ = self.model(self.handler.torchBiAdj_trn)
			itmEmbeds = self.model.getItemFeatsTst()

		for i, batch_data in enumerate(tstLoader):
			usr = batch_data
			numpy_usr = usr.numpy()
			usr = usr.long().cuda()
			if args.zero_shot == 0:
				trnMask = tstLoader.dataset.csrmat[numpy_usr].tocoo()
				trnMask = torch.from_numpy(np.stack([trnMask.row, trnMask.col], axis=0)).long().cuda()
				trnMask = torch.sparse.FloatTensor(trnMask, torch.ones(trnMask.shape[1]).cuda(), torch.Size([usr.shape[0], itmEmbeds.shape[0]]))
				trnMask = trnMask.to_dense()
				allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			else:
				allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0))
			_, topLocs_20 = torch.topk(allPreds, args.topk)
			_, topLocs_40 = torch.topk(allPreds, 40)
			recall_20, ndcg_20 = self.calcRes(topLocs_20.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 20)
			recall_40, ndcg_40 = self.calcRes(topLocs_40.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr, 40)
			epRecall_20 += recall_20
			epNdcg_20 += ndcg_20
			epRecall_40 += recall_40
			epNdcg_40 += ndcg_40
			log('Steps %d/%d: recall@20 = %.2f, ndcg@20 = %.2f, recall@40 = %.2f, ndcg@40 = %.2f          ' % (i, steps, recall_20, ndcg_20, recall_40, ndcg_40), save=False, oneline=True)
		ret = dict()
		ret['Recall@20'] = epRecall_20 / num
		ret['NDCG@20'] = epNdcg_20 / num
		ret['Recall@40'] = epRecall_40 / num
		ret['NDCG@40'] = epNdcg_40 / num
		return ret

	def calcRes(self, topLocs, tstLocs, batIds, topk):
		assert topLocs.shape[0] == len(batIds)
		allRecall = allNdcg = 0
		for i in range(len(batIds)):
			temTopLocs = list(topLocs[i])
			temTstLocs = tstLocs[batIds[i]]
			tstNum = len(temTstLocs)
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, topk))])
			recall = dcg = 0
			for val in temTstLocs:
				if val in temTopLocs:
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum
			ndcg = dcg / maxDcg
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg

def seed_it(seed):
	random.seed(seed)
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	seed_it(421)

	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	coach = Coach(handler)
	coach.run()