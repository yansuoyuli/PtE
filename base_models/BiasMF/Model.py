from statistics import mean
import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np

uniformInit = nn.init.uniform_

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(uniformInit(torch.empty(args.user, args.latdim)))

		self.uBias = nn.Parameter(uniformInit(torch.empty(args.user)))

		self.trans_profile = nn.Sequential(nn.Linear(args.item_feat_dim, args.latdim), nn.ReLU(inplace=True), nn.Linear(args.latdim, args.latdim))
		self.trans_original = nn.Sequential(nn.Linear(args.item_feat_dim, args.latdim), nn.ReLU(inplace=True), nn.Linear(args.latdim, args.latdim))
		
		self.concat_layer = nn.Sequential(nn.Linear(args.latdim * 2, args.latdim), nn.ReLU(inplace=True))

		self.dropout = nn.Dropout(p=args.drop_rate)
	
	def setItemFeatsTrn(self, item_feats):
		self.item_feats_trn = item_feats

	def setItemFeatsTst(self, item_feats):
		self.item_feats_tst = item_feats

	def setItemFeatsTrnProfile(self, item_feats):
		self.item_feats_trn_profile = item_feats

	def setItemFeatsTstProfile(self, item_feats):
		self.item_feats_tst_profile = item_feats

	def setUserFeats(self, user_feats):
		self.user_feats = user_feats

	def predictPairs(self, ancs, itms):
		item_feats = self.trans_original(self.item_feats_trn)
		if args.user_aug == 1:
			user_feats = self.dropout(self.trans_profile(self.user_feats))
			item_feats = self.dropout(self.concat_layer(torch.concat([item_feats, self.trans_profile(self.item_feats_trn_profile)], axis=1)))

		iniUEmbeds = self.uEmbeds
		if args.user_aug == 1:
			iniUEmbeds = self.concat_layer(torch.concat([self.uEmbeds, F.normalize(user_feats)], axis=1))
		ancEmbeds = iniUEmbeds[ancs]
		posEmbeds = F.normalize(item_feats)[itms]
		ancBias = self.uBias[ancs]

		return (ancEmbeds * posEmbeds).sum(-1) + ancBias
	
	def predictAll(self, ancs):
		item_feats = self.trans_original(self.item_feats_tst)
		if args.user_aug == 1:
			user_feats = self.trans_profile(self.user_feats)
			item_feats = self.concat_layer(torch.concat([item_feats, self.trans_profile(self.item_feats_tst_profile)], axis=1))
		
		iniUEmbeds = self.uEmbeds
		if args.user_aug == 1:
			iniUEmbeds = self.concat_layer(torch.concat([self.uEmbeds, F.normalize(user_feats)], axis=1))
		ancEmbeds = iniUEmbeds[ancs]
		ancBias = self.uBias[ancs]

		return ancEmbeds @ F.normalize(item_feats).T + ancBias.view([-1, 1])