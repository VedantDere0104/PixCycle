import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CyclePixModel(BaseModel):
    
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def __init__(self , opt):
        BaseModel.__init__(self , opt)

        self.loss_names = ['D_A' , 'G_A' , 'cycle_A' , 'idt_A' , 'D_B' , 'G_B' , 'cycle_B' , 'idt_B' , 'G_pix_A' , 'D_pix_A']

        visual_names_A = ['real_A' , 'fake_B' , 'rec_A']
        visual_names_B = ['real_B' , 'fake_B' , 'rec_A']

        if self.isTrain and self.opt.lambda_identity > 0.0: 
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')     

        self.visual_names = visual_names_A + visual_names_B
        
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  
            self.model_names = ['G_A', 'G_B']
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.pix_G = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain: 
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.pix_D = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            if opt.lambda_identity > 0.0:
                assert(opt.input_nc == opt.output_nc)
            
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionCycle = torch.nn.L1Loss()
            self.pix2pix = torch.nn.L1Loss()

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters() , self.pix_G), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters() , self.pix_D), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self , input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)

        self.fake_A = self.netG_B(self.fake_B)
        self.rec_B = self.netG_A(self.fake_A)

        self.pix_B = self.pix_G(self.real_A)
        self.pix_B_ = self.pix_G(self.fake_A)
        self.pix_B__ = self.pix_G(self.rec_A)

    def backward_D_basic(self , netD , real , fake):
        real_pred = netD(real)
        loss_D_real = self.criterionGAN(real_pred , True)

        pred_fake = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake , False)

        loss_D = (loss_D_real + loss_D_fake) * 0.5

        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A , self.real_B , fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)        

    def backward_Pix_D(self):
        fake_pB = self.fake_B_pool.query(self.pix_B)
        fake_B_ = self.fake_B_pool.query(self.pix_B_)
        fake_B__ = self.fake_B_pool.query(self.pix_B__)
        self.pix_D_loss = self.backward_D_basic(self.pix_D , self.real_B , fake_pB)
        self.pix_D_loss_ = self.backward_D_basic(self.pix_D , self.real_B , fake_B_)
        self.pix_D_loss__ = self.backward_D_basic(self.pix_D , self.real_B , fake_B__)

        #self.loss_pix_D = self.pix_D_loss + 0.5 * self.pix_D_loss_ + 0.65 * self.pix_D_loss__

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        if lambda_idt > 0:
            
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B) , True)

        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A) , True)

        self.loss_pix_A = self.criterionGAN(self.pix_G(self.real_A) , True)

        self.loss_cycle_A = self.criterionCycle(self.rec_A , self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B , self.real_B) * lambda_B
        self.loss_pix_G = self.criterionCycle(self.fake_pB , self.real_B) 
        self.loss_pix_G_ = self.criterionCycle(self.fake_B_ , self.real_B) * 0.5
        self.loss_pix_G__ = self.criterionCycle(self.fake_B__ , self.real_B) * 0.65

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_pix_A + self.loss_pix_G + self.loss_pix_G_ + self.loss_pix_G__
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A , self.netD_B , self.pix_D] , False)

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

        self.set_requires_grad([self.net_D_A , self.net_D_B , self.pix_D] , True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.backward_Pix_D()
        self.optimizer_D.step()