import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
import torch.utils.data as data
import torch.utils.data as dataloader
import torch
from sklearn.decomposition import PCA

class DataHandler:
	def __init__(self):
		if args.data == 'mind':
			predir = './../../data/mind/'
		elif args.data == 'netflix':
			predir = './../../data/netflix/'
		self.predir = predir
		self.trnfile = predir + 'trnMat_zero.pkl'
		self.tstfile = predir + 'tstMat_zero_.pkl'
		self.maskfile = predir + 'maskMat_zero.pkl'
		self.tstfilezero = predir + 'tstMat_zero_shot.pkl'

		if args.user_aug == 0:
			self.itemfile = predir + 'item_original_features.npy'
		else:
			self.itemfile = predir + 'item_original_features.npy'
			if args.data == 'mind':
				self.itemfile_profile = predir + 'item_profile/item_profile_embeddings.npy'
			elif args.data == 'netflix':
				self.itemfile_profile = predir + 'item_profile/item_profile_embeddings.npy'

		if args.data == 'mind':
			self.userfile = predir + 'user_profile/user_profile_embeddings.npy'

		if args.data == 'netflix':
			#self.userfile = "~data/netflix/user_profile/netflix_final_rlhf_testmaskv1_batch_4_step_2000/item_profile_embeddings.npy"#"~data/netflix/user_profile/netflix_final_rlhf_testmaskv1_step_2000_origin/user_profile_embeddings.npy"#predir + 'user_profile/user_profile_embeddings.npy'
			self.userfile =args.user_aug_path
   
	def loadFeatures(self, filename):
		feats = np.load(filename, allow_pickle=True)
		return torch.tensor(feats).float().cuda(), np.shape(feats)[1]

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat, item_n):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((item_n, item_n))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):
		with open(self.predir + 'item_id_map_train.pkl', 'rb') as f:
			item_id_map_train = pickle.load(f)

		with open(self.predir + 'item_id_map_test.pkl', 'rb') as f:
			item_id_map_test = pickle.load(f)

		if args.zero_shot == 1:
			with open(self.predir + 'item_id_map_zero.pkl', 'rb') as f:
				item_id_map_zero = pickle.load(f)

		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		if args.zero_shot == 1:
			tstMat = self.loadOneFile(self.tstfilezero)
		maskMat = self.loadOneFile(self.maskfile)
		self.trnMat = trnMat
		args.user, args.item_trn = trnMat.shape
		_, args.item_tst = tstMat.shape
		self.torchBiAdj_trn = self.makeTorchAdj(trnMat, args.item_trn)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat, maskMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		self.item_feats, args.item_feat_dim = self.loadFeatures(self.itemfile)
		if args.user_aug == 1:
			self.item_feats_profile, _ = self.loadFeatures(self.itemfile_profile)
		self.item_feats_trn = torch.stack([self.item_feats[i] for i in item_id_map_train.keys()])
		if args.zero_shot == 0:
			self.item_feats_tst = torch.stack([self.item_feats[i] for i in item_id_map_test.keys()])
		else:
			self.item_feats_tst = torch.stack([self.item_feats[i] for i in item_id_map_zero.keys()])
		if args.user_aug == 1:
			self.item_feats_trn_profile = torch.stack([self.item_feats_profile[i] for i in item_id_map_train.keys()])
			if args.zero_shot == 0:
				self.item_feats_tst_profile = torch.stack([self.item_feats_profile[i] for i in item_id_map_test.keys()])
			else:
				self.item_feats_tst_profile = torch.stack([self.item_feats_profile[i] for i in item_id_map_zero.keys()])
		print(self.item_feats_trn.shape)
		print(self.item_feats_tst.shape)
		print("train interactions: ", len(trnMat.row))
		print("test interactions: ", len(tstMat.row))

		self.user_feats, args.user_feat_dim = self.loadFeatures(self.userfile)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item_trn)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat, maskMat):
		self.csrmat = (maskMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)
	
	def __getitem__(self, idx):
		return self.tstUsrs[idx]