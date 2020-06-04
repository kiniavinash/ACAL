import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AcalModel(BaseModel):
    """
    This class implements the ACAL model, for low resource domain adaptation

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    ACAL paper: https://arxiv.org/pdf/1807.00374.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--netM', type=str, default='lenet',
                                help='specify source task specific model architecture [lenet | ... ]')
            parser.add_argument('--source_model', required=True, type=str,
                                help='path to the pre-trained source model')
            parser.add_argument('--lr_task', type=float, default=0.001,
                                help='initial learning rate for task specific model')
        return parser

    def __init__(self, opt):
        """Initialize the ACAL class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'task_A' 'D_B', 'G_B', 'cycle_B', 'task_B']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G_{S->T}), G_B (G_{T->S}), D_A (D_S), D_B (D_T), M_A (M_S), M_B (M_T)
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define discriminators & Task specific models
            self.netM_A = networks.define_M(opt.input_nc, opt.netM, opt.source_model, gpu_ids=self.gpu_ids)
            self.netM_B = networks.define_M(opt.output_nc, opt.netM, None, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionTask = torch.nn.NLLLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_M = torch.optim.Adam(itertools.chain(self.netM_A.parameters(), self.netM_B.parameters()),
                                                lr=opt.lr_task, betas=(0.5, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_M)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.labels_A = input['A_label' if AtoB else 'B_label'].to(self.device)
        self.labels_B = input['B_label' if AtoB else 'A_label'].to(self.device)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

        if self.isTrain:
            self.labelH_real_A = self.netM_A(self.real_A)  # M_A(A)
            self.labelH_fake_A = self.netM_A(self.fake_A)  # M_A(G_A(A))
            self.labelH_real_B = self.netM_B(self.real_B)  # M_B(B)
            self.labelH_fake_B = self.netM_B(self.fake_B)  # M_B(G_B(B))

            self.labelH_rec_A = self.netM_A(self.rec_A)  # M_A(G_B(G_A(A))
            self.labelH_rec_B = self.netM_B(self.rec_B)  # M_B(G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake, netM, labels):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Task-specifc loss
        loss_task = self.criterionTask(netM(real), labels)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D += loss_task  # TODO: is loss_task required here?
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B, self.netM_A, self.labels_A)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A, self.netM_B, self.labels_B)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle task-specific loss
        self.loss_cyc_task_A = self.criterionTask(self.labelH_fake_A, self.labels_B)
        # Backward cycle task-specifc loss
        self.loss_cyc_task_B = self.criterionTask(self.labelH_fake_B, self.labels_A)
        # Forward cycle consistency loss
        self.loss_cycle_A = self.criterionTask(self.labelH_rec_A, self.labels_A) * lambda_A
        # Backward cycle consistency loss
        self.loss_cycle_B = self.criterionTask(self.labelH_rec_B, self.labels_B) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cyc_task_A + self.loss_cyc_task_B + \
                      self.loss_cycle_A + self.loss_cycle_B
        self.loss_G.backward()

    def backward_M_basic(self, label_hat_real, label_real):  # , label_hat_fake,  label_fake):
        """Calculate task loss for the task-specific models

        Parameters
            --- TBD

        Return the task-specific loss.
        We also call loss_M.backward() to calculate the gradients.
        """
        # loss_M_real = self.criterionTask(label_hat_real, label_real)
        # loss_M_fake = self.criterionTask(label_hat_fake, label_fake)
        # loss_M = (loss_M_real + loss_M_fake) * 0.5
        loss_M = self.criterionTask(label_hat_real, label_real)
        loss_M.backward()
        return loss_M

    def backward_M_A(self):
        """Calculate the loss for M_A"""
        self.loss_task_A = self.backward_M_basic(self.labelH_real_A,  self.labels_A)  #, self.labelH_fake_A, self.labels_B)

    def backward_M_B(self):
        """Calculate the loss for M_B"""
        self.loss_task_B = self.backward_M_basic(self.labelH_real_B, self.labels_B)  #, self.labelH_fake_B,  self.labels_A)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.optimizer_M.zero_grad()
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate gradients for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        # M_A and M_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_M.zero_grad()
        self.backward_M_A()
        self.backward_M_B()
        self.optimizer_M.step()
