from torch import nn
import torch.nn.functional as F
from Params import args
import torch

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

		self.trans_profile = nn.Sequential(nn.Linear(args.item_feat_dim, args.latdim), nn.ReLU(inplace=True), nn.Linear(args.latdim, args.latdim))
		self.trans_original = nn.Sequential(nn.Linear(args.item_feat_dim, args.latdim), nn.ReLU(inplace=True), nn.Linear(args.latdim, args.latdim))

		self.concat_layer = nn.Sequential(nn.Linear(args.latdim * 2, args.latdim), nn.ReLU(inplace=True))

		self.embed_perturb = EmbedPerturb(eps=args.eps)

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
	
	def getItemFeatsTst(self):
		if args.user_aug == 0:
			return F.normalize(self.trans_original(self.item_feats_tst))
		else:
			return F.normalize(self.concat_layer(torch.concat([self.trans_original(self.item_feats_tst), self.trans_profile(self.item_feats_tst_profile)], axis=1)))

	def forward(self, adj):
		item_feats = self.trans_original(self.item_feats_trn)
		if args.user_aug == 1:
			user_feats = self.dropout(self.trans_profile(self.user_feats))
			item_feats = self.dropout(self.concat_layer(torch.concat([item_feats, self.trans_profile(self.item_feats_trn_profile)], axis=1)))

		iniEmbeds = torch.concat([self.uEmbeds, F.normalize(item_feats)], axis=0)
		if args.user_aug == 1:
			iniEmbeds = torch.concat([self.concat_layer(torch.concat([self.uEmbeds, F.normalize(user_feats)], axis=1)), F.normalize(item_feats)], axis=0)

		embedsLst = [iniEmbeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		mainEmbeds = sum(embedsLst)

		iniEmbeds1 = torch.concat([self.uEmbeds, F.normalize(item_feats)], axis=0)
		if args.user_aug == 1:
			iniEmbeds1 = torch.concat([self.concat_layer(torch.concat([self.uEmbeds, F.normalize(user_feats)], axis=1)), F.normalize(item_feats)], axis=0)
		embedsLst = [iniEmbeds1]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embeds = self.embed_perturb(embeds)
			embedsLst.append(embeds)
		embedsView1 = sum(embedsLst)

		iniEmbeds2 = torch.concat([self.uEmbeds, F.normalize(item_feats)], axis=0)
		if args.user_aug == 1:
			iniEmbeds2 = torch.concat([self.concat_layer(torch.concat([self.uEmbeds, F.normalize(user_feats)], axis=1)), F.normalize(item_feats)], axis=0)
		embedsLst = [iniEmbeds2]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embeds = self.embed_perturb(embeds)
			embedsLst.append(embeds)
		embedsView2 = sum(embedsLst)

		return mainEmbeds[:args.user], F.normalize(item_feats), embedsView1[:args.user], embedsView1[args.user:], embedsView2[:args.user], embedsView2[args.user:]

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

class EmbedPerturb(nn.Module):
    def __init__(self, eps):
        super(EmbedPerturb, self).__init__()
        self.eps = eps

    def forward(self, embeds):
        noise = (F.normalize(torch.rand(embeds.shape).cuda(), p=2) * torch.sign(embeds)) * self.eps
        embeds = embeds + noise
        return embeds
