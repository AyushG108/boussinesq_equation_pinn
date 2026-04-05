class Net:
    def __init__(self):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.model = NN().to(device)

        # Computational domain
        self.h = 0.1
        self.k = 0.05
        x = torch.arange(-20, 20 + self.h, self.h)
        t = torch.arange(-5, 5 + self.k, self.k)

        self.X = torch.stack(torch.meshgrid(x, t)).reshape(2, -1).T

        bc1 = torch.stack(torch.meshgrid(x[0], t)).reshape(2, -1).T
        bc2 = torch.stack(torch.meshgrid(x[-1], t)).reshape(2, -1).T
        ic = torch.stack(torch.meshgrid(x, t[0])).reshape(2, -1).T
        self.X_train = torch.cat([bc1, bc2, ic])

        y_bc1 = torch.zeros(len(bc1))
        y_bc2 = torch.zeros(len(bc2))

        hyp_ic = torch.cosh((ic[:, 0] - 5 * 1.414) / 2)
        y_ic = 1 / (2 * hyp_ic)

        self.y_train = torch.cat([y_bc1, y_bc2, y_ic])
        self.y_train = self.y_train.unsqueeze(1)

        self.X = self.X.to(device)
        self.y_train = self.y_train.to(device)
        self.X_train = self.X_train.to(device)
        self.X.requires_grad = True

        # Optimizer setting
        self.adam = torch.optim.Adam(self.model.parameters())
        # Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS)
        self.optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            lr=1.0,
            max_iter=170,
            max_eval=170,
            history_size=50,
            tolerance_grad=1e-7,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe"
        )

        self.criterion = torch.nn.MSELoss()
        self.iter = 1

    def loss_func(self):
        self.adam.zero_grad()
        self.optimizer.zero_grad()

        y_pred = self.model(self.X_train)
        loss_data = self.criterion(y_pred, self.y_train)

        u = self.model(self.X)

        # Calculate gradients
        grad_u = torch.autograd.grad(
            u,
            self.X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

        dudt = grad_u[:, 1]
        dudx = grad_u[:, 0]
        # Calculate second derivatives of time
        dudtt = torch.autograd.grad(
            dudt,
            self.X,
            grad_outputs=torch.ones_like(dudt),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]

        # Calculate second derivatives
        dudxx = torch.autograd.grad(
            dudx,
            self.X,
            grad_outputs=torch.ones_like(dudx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
        # Calculate 3rd derivatives
        dudxxx = torch.autograd.grad(
            dudxx,
            self.X,
            grad_outputs=torch.ones_like(dudxx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        # Calculate 4th derivatives
        dudxxxx = torch.autograd.grad(
            dudxxx,
            self.X,
            grad_outputs=torch.ones_like(dudxxx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        # Calculate second derivatives of u^2
        duu_dx = torch.autograd.grad(
            u,
            self.X,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0][:, 1]

        duu_dxx = torch.autograd.grad(
            duu_dx,
            self.X,
            grad_outputs=torch.ones_like(duu_dx),
            create_graph=True,
            retain_graph=True
        )[0][:, 0]

        loss_pde = self.criterion(dudtt, dudxx + duu_dxx + dudxxxx)

        loss = loss_pde + loss_data
        loss.backward()

        if self.iter % 10 == 0:
            print(self.iter, loss.item())
        self.iter = self.iter + 1

        return loss

    def train(self):
        self.model.train()
        for i in range(85):
            self.adam.step(self.loss_func)
        self.optimizer.step(self.loss_func)

    def eval_(self):
        self.model.eval()

# training
net = Net()
net.train()