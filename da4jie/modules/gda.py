import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# Note: FeatureNet, PredNet, GNet, GraphDNet are external model components
# from model.modules import *
import pickle
import random
# from visdom import Visdom  # optional visualization dependency

# ======================================================================================================================
def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def flat(x):
    n, m = x.shape[:2]
    return x.reshape(n * m, *x.shape[2:])


def write_pickle(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)

# ======================================================================================================================

class GDA():
    def __init__(self, opt):
        self.opt = opt
        self.device = opt.device
        self.batch_size = opt.batch_size
        self.num_domain = opt.num_domain
        self.train_log = self.opt.outf + "/loss.log"
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        with open(self.train_log, 'w') as f:
            f.write("log start!\n")

        self.all_loss_G = 0
        self.all_loss_D = 0
        self.all_loss_E_gan = 0
        self.all_loss_E_pred = 0
        self.epoch = 0

        # self.__init_visdom__()
        self.__set_num_domain__(opt.num_domain)

        # temporarily don't consider target domain in the middle
        # mask_list = [1] * opt.num_source + [0] * opt.num_target
        mask_list = np.zeros(opt.num_domain)
        mask_list[opt.source_domain] = 1
        self.domain_mask = torch.IntTensor(mask_list).to(opt.device)  # not sure if device is needed

        self.netE = FeatureNet(opt).to(opt.device)
        self.netF = PredNet(opt).to(opt.device)
        # G now changed to a real network
        self.netG = GNet(opt).to(opt.device)
        if opt.load_g:
            self.netG.load_state_dict(torch.load(self.opt.loadf + "/GDA_" + self.netG.__class__.__name__ + "_pretrain"))

        self.netD = GraphDNet(opt).to(opt.device)   
        # self.netD = ResGraphDNet(opt).to(opt.device)
        self.nets = [self.netE, self.netF, self.netD, self.netG]

        self.__init_weight__()

        # optimizer
        EF_parameters = list(self.netE.parameters()) + list(self.netF.parameters())
        self.optimizer_EF= optim.Adam(EF_parameters, lr=opt.lr_e, betas=(opt.beta1, 0.999))
        # try 3* learning rate for G
        self.optimizer_D = optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
        
        # G parameter is special!
        G_params = list(self.netG.parameters()) + [self.netG.bias] + [self.netG.weight]
        self.optimizer_G = optim.Adam(G_params, lr=opt.lr_g, betas=(opt.beta1, 0.999))

        # scheduler
        # be sure about the gamma !
        self.lr_scheduler_EF = lr_scheduler.ExponentialLR(optimizer=self.optimizer_EF, gamma=0.5 ** (1/100))
        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.optimizer_D, gamma=0.5 ** (1/100))
        self.lr_scheduler_G = lr_scheduler.ExponentialLR(optimizer=self.optimizer_G, gamma=0.5 ** (1 / 100))
        self.lr_schedulers = [self.lr_scheduler_EF, self.lr_scheduler_D, self.lr_scheduler_G]
    
    def __init_visdom__(self):
        self.env = Visdom(port=2000)
        self.pane_D = self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_D]),
            opts=dict(
                title='loss D on epochs',
                plotly={'legend':{'x':200, 'y':0}}
            )
        )
        self.pane_E_pred = self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_E_pred]),
            opts=dict(title='loss E pred on epochs')
        )
        self.pane_G = self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_G]),
            opts=dict(title='loss G on epochs')
        )

        # self.pane_G_bias = self.env.line(
        #     X=np.array([self.epoch]),
        #     Y=np.array([to_np(self.netG.bias)]),
        #     opts=dict(title='G bias on epochs')
        # )

        # self.pane_G_weight = self.env.line(
        #     X=np.array([self.epoch]),
        #     Y=np.array([to_np(self.netG.weight)]),
        #     opts=dict(title='G weight on epochs')
        # )

    def __vis_loss__(self):
        self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_D]),
            win=self.pane_D,
            update='append'
        )
        self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_E_pred]),
            win=self.pane_E_pred,
            update='append'
        )
        self.env.line(
            X=np.array([self.epoch]),
            Y=np.array([self.all_loss_G]),
            win=self.pane_G,
            update='append'
        )

        # self.env.line(
        #     X=np.array([self.epoch]),
        #     Y=np.array([to_np(self.netG.bias)]),
        #     win=self.pane_G_bias,
        #     update='append'
        # )

        # self.env.line(
        #     X=np.array([self.epoch]),
        #     Y=np.array([to_np(self.netG.weight)]),
        #     win=self.pane_G_weight,
        #     update='append'
        # )

    def set_data_stats(self, dm, ds):
        self.data_m, self.data_s = dm, ds

    def __init_weight__(self):
        for net in self.nets:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01)
                    nn.init.constant_(m.bias, val=0)
    
    def __set_num_domain__(self, num):
        # t is domain index: normalized to [0,1]
        self.t = np.linspace(0, 1, num).astype(np.float32)
        self.t_var = to_tensor(self.t, self.device)
        # # z is domain class (0,1,2,...) will be used by some adaptation methods
        # self.z = np.arange(num).astype(np.int64)
        # self.z_var = to_tensor(self.z, self.device)

    def __log_write__(self, loss_msg):
        print(loss_msg)
        with open(self.train_log, 'a') as f:
            f.write(loss_msg + "\n")

    def learn(self, epoch, dataloader):
        self.epoch = epoch

        self.all_loss_G = 0
        self.all_loss_D = 0
        self.all_loss_E_gan = 0
        self.all_loss_E_pred = 0


        for data in dataloader:
            self.__set_input__(data)
            
            self.__forward__()

            # # optimization
            # loss G, will be terminated after half training
            # if epoch < self.opt.num_epoch * 0.67:
            loss_G = self.__optimize_G__()
            self.all_loss_G += loss_G

            loss_D = self.__optimize_D__()
            self.all_loss_D += loss_D
            loss_E_gan, loss_E_pred = self.__optimize_EF__()
            self.all_loss_E_gan += loss_E_gan
            self.all_loss_E_pred += loss_E_pred

        # temporary use it
        if (epoch + 1) % 5 == 0:
            # print("loss D: {:.4f}, loss E gan: {:.4f}, loss E pre: {:.4f}, loss G: {:.4f}".format(all_loss_D, all_loss_E_gan, all_loss_E_pred, all_loss_G))
            self.__log_write__("loss D: {:.4f}, loss E gan: {:.4f}, loss E pre: {:.4f}, loss G: {:.4f}".format(
                self.all_loss_D, 
                self.all_loss_E_gan, 
                self.all_loss_E_pred, 
                self.all_loss_G)
            )
        if self.opt.visualize:
            if epoch == 0:
                self.__init_visdom__()
            else:
                self.__vis_loss__()


        # learning rate decay
        for lr_scheduler in self.lr_schedulers:
            lr_scheduler.step()
    
    def __set_input__(self, data):
        """
        :param
            x_seq: Number of domain x Batch size x Data dim
            y_seq: Number of domain x Batch size x Label dim
            t_seq: Number of domain x Batch size x Number of vertices (domains)
            domain_seq: Number of domain x Batch size x domain dim (1)
        """
        x_seq, y_seq = [d[0][None, :, :] for d in data], [d[1][None, :] for d in data]
        self.x_seq = torch.cat(x_seq, 0).to(self.device)
        self.y_seq = torch.cat(y_seq, 0).to(self.device)

        # # t seq for the domain index normalized in [0, 1]
        # self.T, self.B = self.x_seq.shape[:2]
        # self.t_seq = to_tensor(np.zeros((self.T, self.B, 1), dtype=np.float32), self.device) + self.t_var.reshape(self.T, 1, 1)

        # domain seq
        domain_seq = [d[2][None, :] for d in data]
        self.domain_seq = torch.cat(domain_seq, 0)# .to(self.device)

        # one_hot for vertex
        t_seq = [torch.nn.functional.one_hot(d[2], self.num_domain) for d in data]
        self.t_seq = torch.cat(t_seq, 0).reshape(self.num_domain, self.batch_size, -1).to(self.device)

    def __set_requires_grad__(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def __forward__(self):
        for net in self.nets:
            # print("=====")
            # print(net)
            net.train()

        self.z_seq = self.netG(self.t_seq)
        self.e_seq = self.netE(self.x_seq, self.z_seq)
        self.f_seq = self.netF(self.e_seq)
        self.d_seq = self.netD(self.e_seq)

        # this is the d loss, still not backward yet
        self.loss_D = self.__loss_D__(self.d_seq)

    # this is an iterator for edge sampling
    # may be too slow
    # class basic_edge_sampler():
    #     # this is a basic sampler that will use several nodes' subgraph as the sampled edge
    #     def __iter__(self, sample_v, num_v_all):
    #         self.sample_v = sample_v
    #         self.num_v_all = num_v_all
    #         self.i = 0
    #         self.j = 0
    #         self.sub_graph = self.__sub_graph__()
    #         return self

    #     def __next__(self):
    #         self.j += 1
    #         if self.j >= self.sample_v:
    #             self.i += 1
    #             if self.i == self.sample_v - 1:
    #                 raise StopIteration
    #             else:
    #                 self.j = self.i + 1
            
    #         return self.sub_graph[self.i], self.sub_graph[self.j]

    #     def __sub_graph__(self):
    #         return np.random.choice(self.num_v_all, size=self.sample_v, replace=False)
    def __rand_walk__(self, vis, left_nodes):
        chain_node = []
        node_num = 0
        # choose node
        node_index = np.where(vis == 0)[0]
        # print(node_index)
        st = np.random.choice(node_index)
        # print(st)
        vis[st] = 1
        chain_node.append(st)
        left_nodes -= 1
        node_num += 1
        
        cur_node = st
        while left_nodes > 0:
            nx_node = -1

            node_to_choose = np.where(vis == 0)[0]
            num = node_to_choose.shape[0]
            node_to_choose = np.random.choice(node_to_choose, num, replace=False)

            for i in node_to_choose:
                if cur_node != i:
                    # have an edge and doesn't visit
                    if self.opt.A[cur_node][i] and not vis[i]:
                        nx_node = i
                        vis[nx_node] = 1
                        chain_node.append(nx_node)
                        left_nodes -= 1
                        node_num += 1
                        break
            if nx_node >= 0:
                cur_node = nx_node
            else:
                break
        # print("===chain===")
        # print(chain_node)
        # print(node_num)
        return chain_node, node_num

    def __sub_graph__(self, my_sample_v):
        # play
        if np.random.randint(0,2) == 0:
            return np.random.choice(self.num_domain, size=my_sample_v, replace=False)
        
        # subsample a chain (or multiple chains in graph)
        # Todo: 需要验证指针传递还是字符串传递！！
        # left_nodes = self.opt.sample_v
        left_nodes = my_sample_v
        choosen_node = []
        vis = np.zeros(self.num_domain)
        while left_nodes > 0:
            chain_node, node_num = self.__rand_walk__(vis, left_nodes) 
            # vis = np.zeros(self.num_domain)
            choosen_node.extend(chain_node)
            left_nodes -= node_num
        
        # print("==choosen==")
        # print(choosen_node)
        return choosen_node
        # return np.random.choice(self.num_domain, size=self.opt.sample_v, replace=False)

    def __optimize_G__(self):
        self.netG.train()
        self.netD.eval(), self.netE.eval(), self.netF.eval()

        self.optimizer_G.zero_grad()

        criterion = nn.BCEWithLogitsLoss()

        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v_g)
        errorG = torch.zeros((1,)).to(self.device)
        # errorG_connected = torch.zeros((1,)).to(self.device)
        # errorG_disconnected = torch.zeros((1,)).to(self.device)
        # count_connected = 0
        # count_disconnected = 0

        sample_v = self.opt.sample_v_g
        # train_z_seq = self.z_seq.reshape()
        
        for i in range(sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, sample_v):
                v_j = sub_graph[j]
                # label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                # label = torch.full((self.batch_size,), self.opt.A[v_i][v_j], device=self.device)
                label = torch.tensor(self.opt.A[v_i][v_j]).to(self.device)
                # dot product
                output = self.netG.weight * (self.z_seq[v_i * self.batch_size] * self.z_seq[v_j * self.batch_size]).sum() + self.netG.bias
                errorG += criterion(output, label)

                # for training of balancing
                # if self.opt.A[v_i][v_j]: # connected
                #     errorG_connected += criterion(output, label)
                #     count_connected += 1
                # else:
                #     errorG_disconnected += criterion(output, label)
                #     count_disconnected += 1

        errorG /= (sample_v * (sample_v - 1) / 2)
        # errorG = 0.5 * (errorG_connected / count_connected + errorG_disconnected / count_disconnected)
        # alpha = self.opt.g_alpha
        # errorG = alpha * errorG_connected / count_connected + (1 - alpha) * errorG_disconnected / count_disconnected
        # errorG /= self.batch_size
        # errorG *= self.num_domain
        # errorG = errorG / self.batch_size / (sample_v * (sample_v - 1) / 2)
        # errorG = errorG / (sample_v * (sample_v - 1) / 2)
        # make regularization
        # errorG += 0.005 * (self.z_seq * self.z_seq).sum() / self.batch_size
        # errorG += 0.005 * nn.MSELoss()(self.z_seq, torch.zeros_like(self.z_seq)) / self.batch_size

        errorG.backward(retain_graph=True)
        
        self.optimizer_G.step()
        return errorG.item()

        
    def __optimize_D__(self):
        self.netD.train()
        self.netG.eval(), 
        self.netE.eval(), self.netF.eval()

        self.optimizer_D.zero_grad()

        # backward process:
        self.loss_D.backward(retain_graph=True)

        self.optimizer_D.step()
        return self.loss_D.item()

    def __loss_D__(self, d):
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.BCELoss()
        # criterion = nn.MSELoss()

        # random pick subchain and optimize the D
        # balance coefficient is calculate by pos/neg ratio

        sub_graph = self.__sub_graph__(my_sample_v=self.opt.sample_v)

        errorD_connected = torch.zeros((1,)).to(self.device)
        errorD_disconnected = torch.zeros((1,)).to(self.device)

        count_connected = 0
        count_disconnected = 0
        
        for i in range(self.opt.sample_v):
            v_i = sub_graph[i]
            for j in range(i + 1, self.opt.sample_v):
            # self loop version:
            # for j in range(i, self.opt.sample_v):
                v_j = sub_graph[j]
                
                label = torch.full((self.batch_size,), self.opt.A[v_i][v_j], device=self.device)
                # dot product
                if v_i == v_j:
                    # be careful about the index range!
                    idx = torch.randperm(self.batch_size)
                    output = (d[v_i][idx] * d[v_j]).sum(1)
                else:
                    output = (d[v_i] * d[v_j]).sum(1)

                # try
                # output = torch.clamp(output, 0, 1)

                if self.opt.A[v_i][v_j]: # connected
                    errorD_connected += criterion(output, label)
                    count_connected += 1
                else:
                    errorD_disconnected += criterion(output, label)
                    count_disconnected += 1

        # all_error_count = count_disconnected + count_connected
        errorD = 0.5 * (errorD_connected / count_connected + errorD_disconnected / count_disconnected)
        # print(errorD.item())
        # this is a loss balance
        return errorD * self.num_domain


    def __optimize_EF__(self):
        self.netD.eval(), self.netG.eval()
        self.netE.train(), self.netF.train()

        # self.__set_requires_grad__(self.netD, False)
        self.optimizer_EF.zero_grad()

        loss_E_gan = - self.loss_D

        y_seq_source = self.y_seq[self.domain_mask == 1]
        f_seq_source = self.f_seq[self.domain_mask == 1]

        loss_E_pred = F.nll_loss(flat(f_seq_source), flat(y_seq_source))
        
        loss_E = loss_E_gan * self.opt.lambda_gan + loss_E_pred
        loss_E.backward()

        self.optimizer_EF.step()

        return loss_E_gan.item(), loss_E_pred.item()



    def save(self):
        if not os.path.exists(self.opt.outf):
            os.mkdir(self.opt.outf)
        for net in self.nets:
            torch.save(net.state_dict(), self.opt.outf + "/GDA_" + net.__class__.__name__)
    
    def load(self):
        for net in self.nets:
            net.load_state_dict(torch.load(self.opt.loadf + "/GDA_" + net.__class__.__name__))

    # def __set_data_stats__(self, dm, ds):
    #     self.data_m, self.data_s = dm, ds

    def test(self, epoch, dataloader):
        # pass
        for net in self.nets:
            net.eval()
        
        # now is the embedding printing version
        acc_curve = []
        l_x = []
        # l_y = []
        l_domain = []
        # l_prob = []
        l_label = []
        l_encode = []
        l_decode = []
        # l_z_seq = []
        # big change on l_z_seq !!!!
        # z_seq = 
        z_seq = 0

        for data in dataloader:
            self.__set_input__(data)

            # forward
            with torch.no_grad():
                z_seq = self.netG(self.t_seq)
                e_seq = self.netE(self.x_seq ,z_seq)
                f_seq = self.netF(e_seq)
                g_seq = torch.argmax(f_seq.detach(), dim=2)  # class of the prediction
                d_seq = self.netD(e_seq)

            acc_curve.append(g_seq.eq(self.y_seq).to(torch.float).mean(-1, keepdim=True))

            if self.opt.normalize_domain:
                # still working on normalize domain
                # pass
                x_np = to_np(self.x_seq)
                for i in range(len(x_np)):
                    x_np[i] = x_np[i] * self.data_s[i] + self.data_m[i]
                l_x.append(x_np)
            else:
                l_x.append(to_np(self.x_seq))

            # l_y.append(to_np(y_seq))
            # l_z_seq.append(to_np(z_seq))
            l_domain.append(to_np(self.domain_seq))
            # l_prob.append(to_np(self.f_seq))
            l_encode.append(to_np(e_seq))
            l_decode.append(to_np(d_seq))
            l_label.append(to_np(g_seq))

        x_all = np.concatenate(l_x, axis=1)
        e_all = np.concatenate(l_encode, axis=1)
        decode_all = np.concatenate(l_decode, axis=1)
        # y_all = np.concatenate(l_y, axis=1)
        domain_all = np.concatenate(l_domain, axis=1)
        # prob_all = np.concatenate(l_prob, axis=1)
        label_all = np.concatenate(l_label, axis=1)

        # print(np.asarray(l_z_seq).shape)
        z_seq = to_np(z_seq)
        z_seq_all = z_seq[0:self.batch_size * self.num_domain:self.batch_size,:]
        # print(z_seq_all.shape)
        # z_seq_all = np.concatenate(l_z_seq, axis=1)

        # print(z_seq_all.shape)
        # print(label_all.shape)
        # print(x_all.shape)


        d_all = dict()

        d_all['data'] = flat(x_all)
        # d_all['gt'] = flat(y_all)
        d_all['domain'] = flat(domain_all)
        # d_all['prob'] = flat(prob_all)
        d_all['label'] = flat(label_all)
        d_all['encodeing'] = flat(e_all)
        d_all['decodeing'] = flat(decode_all)
        d_all['z'] = z_seq_all
        d_all['g_bias'] = to_np(self.netG.bias)
        d_all['g_weight'] = to_np(self.netG.weight)

        acc = to_np(torch.cat(acc_curve, 1).mean(-1))
        test_acc = (acc.sum() - acc[self.opt.source_domain].sum()) / (self.opt.num_target) * 100
        acc_msg = '[Test][{}] Accuracy: total average {:.1f}, test acc: {:.1f}, in each domain {}'.format(epoch, acc.mean() * 100, test_acc, np.around(acc * 100, decimals=1))
        # print(acc_msg)
        self.__log_write__(acc_msg)

        d_all['acc_msg'] = acc_msg

        write_pickle(d_all, self.opt.outf + '/' + str(epoch) + '_pred.pkl')

