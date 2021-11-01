-- usage example: DATA_ROOT=/path/to/data/ which_direction=BtoA name=expt1 th train.lua 
--
-- code derived from https://github.com/soumith/dcgan.torch
--

require 'torch'
require 'nn'
require 'optim'
util = paths.dofile('util/util.lua')
require 'image'
require 'models'
require 'table'
require 'models_grad1'
require 'cunn'
unpack=table.unpack

--set the parameters
opt = {
   DATA_ROOT = '',         -- path to images (should have subfolders 'train', 'val', etc)
   batchSize = 8,          -- # images in batch
   loadSize = 128,         -- scale images to this size
   fineSize = 128,         --  then crop to this size
   ngf = 16,               -- #  of gen filters in first conv layer
   ndf = 64,               -- #  of discrim filters in first conv layer
   input_nc = 3,           -- #  of input image channels
   output_nc = 3,          -- #  of output image channels
   niter = 2000,            -- #  of iter at starting learning rate
   niter_decay=100,          -- # of iter to linearly decay learning rate to zero
   lr = 0.0002,            -- initial learning rate for adam
   beta1 = 0.5,            -- momentum term of adam
   ntrain = math.huge,     -- #  of examples per epoch. math.huge for full dataset
   flip = 1,               -- if flip the images for data argumentation
   display = 1,            -- display samples while training. 0 = false
   display_id = 10,        -- display window id.
   display_plot = 'errL1',    -- which loss values to plot over time. Accepted values include a comma seperated list of: errL1, errG, and errD
   gpu = 1,                -- gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   name = '',              -- name of the experiment, should generally be passed on the command line
   which_direction = 'BtoA',    -- AtoB or BtoA
   phase = 'train',             -- train, val, test, etc
   preprocess = 'regular',      -- for special purpose preprocessing, e.g., for colorization, change this (selects preprocessing functions in util.lua)
   nThreads = 2,                -- # threads for loading data
   save_epoch_freq = 1,        -- save a model every save_epoch_freq epochs (does not overwrite previously saved models)
   save_latest_freq = 1000,     -- save the latest model every latest_freq sgd iterations (overwrites the previous latest model)
   print_freq = 1,             -- print the debug information every print_freq iterations
   display_freq = 100,          -- display the current results every display_freq iterations
   save_display_freq = 500,    -- save the current display of results every save_display_freq_iterations
   continue_train=0,            -- if continue training, load the latest model: 1: true, 0: false
   serial_batches = 0,          -- if 1, takes images in order to make batches, otherwise takes them randomly
   serial_batch_iter = 1,       -- iter into serial image list
   checkpoints_dir = './checkpoints', -- models are saved here
   cudnn = 1,                         -- set to 0 to not use cudnn
   condition_GAN = 0,                 -- set to 0 to use unconditional discriminator
   use_GAN = 1,                       -- set to 0 to turn off GAN term
   use_L1 = 1,                        -- set to 0 to turn off L1 term
   which_model_netD = 'basic', -- selects model to use for netD
   which_model_netRevG='new_model',
   which_model_netG='new_model_10',
n_layers_D = 0,             -- only used if which_model_netD=='n_layers'
                 -- weight on L1 term in objective
 lambdagan = 0.01,               -- weight on L1 term in objective   --.01
 lambdagrad = .1,               -- weight on L1 term in objective   --.1
lambdacnn = 1,                  -- 1      with new_model_10
}
--cutorch.setDevice(opt.gpu)
-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)

local input_nc = opt.input_nc
local output_nc = opt.output_nc
-- translation direction
local idx_A = nil
local idx_B = nil

if opt.which_direction=='AtoB' then
    idx_A = {1, input_nc}
    idx_B = {input_nc+1, input_nc+output_nc}
elseif opt.which_direction=='BtoA' then
    idx_A = {input_nc+1, input_nc+output_nc}
    idx_B = {1, input_nc}
else
    error(string.format('bad direction %s',opt.which_direction))
end

if opt.display == 0 then opt.display = false end

opt.manualSeed = torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setdefaulttensortype('torch.FloatTensor')

-- create data loader
local data_loader = paths.dofile('data/data.lua')
print('#threads...' .. opt.nThreads)
local data = data_loader.new(opt.nThreads, opt)
print("Dataset Size: ", data:size())
tmp_d, tmp_paths = data:getBatch()

-----weight initialization
local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end


local ndf = opt.ndf
local ngf = opt.ngf
local real_label = 1
--local real_label_new=0.9
local fake_label = 0


--- set the model architectures for generator, discriminator and reblurring module
function defineG(input_nc, output_nc, ngf)
    local netG = nil

    if     opt.which_model_netG == "encoder_decoder" then netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netG == "new_model_10" then netG = create_cnn_10(input_nc, output_nc, ngf)
    elseif opt.which_model_netG=="new_model" then netG=create_cnn(input_nc,output_nc,ngf)
    elseif opt.which_model_netG=="new_model_circles" then netG=create_cnn_circles(input_nc,output_nc,ngf)
    else error("unsupported netG model")
    end
   
    netG:apply(weights_init)
  
    return netG
end
function defineRevG(input_nc, output_nc, ngf)
    local Rev_netG = nil

    if     opt.which_model_netRevG == "encoder_decoder" then Rev_netG = defineG_encoder_decoder(input_nc, output_nc, ngf)
    elseif opt.which_model_netRevG=="new_model" then Rev_netG=create_cnn1(input_nc,output_nc,ngf)
    else error("unsupported netG model")
    end
   
    Rev_netG:apply(weights_init)
  
    return Rev_netG
end

function defineD(input_nc, output_nc, ndf)
    local netD = nil
    if opt.condition_GAN==1 then
        input_nc_tmp = input_nc
    else
        input_nc_tmp = 0 -- only penalizes structure in output channels
    end
    
    if     opt.which_model_netD == "basic" then netD = defineD_basic(input_nc_tmp, output_nc, ndf)
    elseif opt.which_model_netD == "n_layers" then netD = defineD_n_layers(input_nc_tmp, output_nc, ndf, opt.n_layers_D)
    else error("unsupported netD model")
    end
    
    netD:apply(weights_init)
    
    return netD
end


-- load saved models and finetune
if opt.continue_train == 1 then
   print('loading previously trained netG...')
   netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), opt)
   print('loading previously trained netD...')
   netD = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), opt)
   print('loading previously trained netD...')
   Rev_netG = util.load(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_RevG.t7'), opt)
else
  print('define model netG...')
  netG = defineG(input_nc, output_nc, ngf)
  print('define model netD...')
  netD = defineD(input_nc, output_nc, ndf)
  Rev_netG=defineRevG(input_nc, output_nc, ngf)
   --model1=create_sml_model1(3,3)
end
--create the dowsampling and gradient extraction modules for 5 scale factors
model=create_grad_model5(3,3)
model1=create_grad_model1(3,3)
model2=create_grad_model2(3,3)
model3=create_grad_model3(3,3)
model4=create_grad_model4(3,3)
print(netG)
print(netD)
print(Rev_netG)

--- define the criterion for each models
local criterion = nn.BCECriterion() --- discriminator
-- l1 criterion for grad modules
local criterionA = nn.AbsCriterion()  
local criterionA1 = nn.AbsCriterion()
local criterionA2 = nn.AbsCriterion()
local criterionA3 = nn.AbsCriterion()
local criterionA4 = nn.AbsCriterion()
--MSE criterion for reblur module
local Rev_criterion=nn.MSECriterion()
---------------------------------------------------------------------------
optimStateG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateRevG = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}
optimStateD = {
   learningRate = opt.lr,
   beta1 = opt.beta1,
}

------------------------------------------------Decaying Learning Rate Function-----------------
------------------------------------------------------------------------------------------------
function UpdateLearningRate(opt)
  local lrd = opt.lr / opt.niter_decay
  local old_lr = self.optimStateD['learningRate']
  local lr =  old_lr - lrd
  optimStateD['learningRate'] = lr
  optimStateRevG['learningRate'] = lr
  optimStateG['learningRate'] = lr
  print(('update learning rate: %f -> %f'):format(old_lr, lr))
end

------------------------------------------------------------------
------ Define the data for training 
local real_A = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize) --input blurred
--downsampled input
local real_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local real_B1 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/2, opt.fineSize/2) 
local real_B2 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/4, opt.fineSize/4)
local real_B3 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/8, opt.fineSize/8)
local real_B4 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/16, opt.fineSize/16)

local real_B1o = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/2, opt.fineSize/2)
local real_B2o = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/4, opt.fineSize/4)
local real_B3o = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/8, opt.fineSize/8)
local real_B4o = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/16, opt.fineSize/16)
local real_Bo = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)

--reblurred data
local Rev_fake_B = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize)

local real_B_new = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize) ---clean image

--generated data and downsampled versions
local fake_B = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize)
local fake_B1 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/2, opt.fineSize/2)
local fake_B2 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/4, opt.fineSize/4)
local fake_B3 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/8, opt.fineSize/8)
local fake_B4 = torch.Tensor(opt.batchSize, output_nc, opt.fineSize/16, opt.fineSize/16)

local fake_B_new = torch.Tensor(opt.batchSize, output_nc, opt.fineSize, opt.fineSize) --generated op

--catenated data for conditional GAN
local real_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)
local fake_AB = torch.Tensor(opt.batchSize, output_nc + input_nc*opt.condition_GAN, opt.fineSize, opt.fineSize)

local errD, errG, errL1,errRevG= 0, 0, 0,0
local epoch_tm = torch.Timer()
local tm = torch.Timer()
local data_tm = torch.Timer()

    --Rev_df_do_new = torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize):fill(0)
   -- Rev_df_do=torch.Tensor(opt.batchSize, input_nc, opt.fineSize, opt.fineSize):fill(0)
----------------------------------------------------------------------------

if opt.gpu > 0 then
   print('transferring to gpu...')
   require 'cunn'
   cutorch.setDevice(opt.gpu)
   real_A = real_A:cuda();
   real_B = real_B:cuda(); fake_B = fake_B:cuda();
   fake_B1 = fake_B1:cuda();
   fake_B2 = fake_B2:cuda();
   fake_B3 = fake_B3:cuda();
   fake_B4 = fake_B4:cuda();
   real_B1 = real_B1:cuda();
   real_B2= real_B2:cuda();
   real_B3 = real_B3:cuda();
   real_B4 = real_B4:cuda();
   real_AB = real_AB:cuda(); fake_AB = fake_AB:cuda();
   real_B_new=real_B_new:cuda()
   fake_B_new=fake_B_new:cuda()
   Rev_fake_B=Rev_fake_B:cuda()
   if opt.cudnn==1 then
      netG = util.cudnn(netG); netD = util.cudnn(netD);Rev_netG=util.cudnn(Rev_netG)
      model=util.cudnn(model);model1=util.cudnn(model1);model2=util.cudnn(model2);model3=util.cudnn(model3);model4=util.cudnn(model4);
   end
   netD:cuda()  netG:cuda() criterion:cuda() criterionA:cuda() criterionA1:cuda() criterionA2:cuda() criterionA3:cuda() criterionA4:cuda() Rev_netG:cuda() Rev_criterion:cuda() model:cuda()  model1:cuda() model2:cuda() model3:cuda() model4:cuda() --model1:cuda()
   --Rev_df_do_new=Rev_df_do_new:cuda()
   --Rev_df_do=Rev_df_do:cuda()
   print('done')
else
	print('running model on CPU')
end


local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()
local parametersRevG,gradParametersRevG=Rev_netG:getParameters()



if opt.display then disp = require 'display' end


function createRealFake()
    -- load real
    data_tm:reset(); data_tm:resume()
    local real_data, data_path = data:getBatch() --fetch batch of data
    data_tm:stop()
    
    real_A:copy(real_data[{ {}, idx_A, {}, {} }])  ---blur input
    

    real_B_new:copy(real_data[{ {}, idx_B, {}, {} }]) -------------clean input-------------
    ---blur data gradients
    real_Bo=model:forward(real_A)  
    real_B:copy(real_Bo)
    real_B1o=model1:forward(real_A)
    real_B1:copy(real_B1o) 
    real_B2o=model2:forward(real_A)
    real_B2:copy(real_B2o)
    real_B3o=model3:forward(real_A) 
    real_B3:copy(real_B3o)
    real_B4o=model4:forward(real_A)
    real_B4:copy(real_B4o)

    if opt.condition_GAN==1 then
        real_AB = torch.cat(real_A,real_B_new,2)
    else
        real_AB = real_B_new -- unconditional GAN, only penalizes structure in B
    end
    
    -- create fake (generated data)
    fake_B_new = netG:forward(real_A)    ----generator output
    --generated data gradients
    fake_B=model:forward(fake_B_new)
    fake_B1=model1:forward(fake_B_new)
    fake_B2=model2:forward(fake_B_new)
    fake_B3=model3:forward(fake_B_new)
    fake_B4=model4:forward(fake_B_new)

    Rev_fake_B=Rev_netG:forward(fake_B_new)   --- reblurred revgenerator output
    
    
    if opt.condition_GAN==1 then
        fake_AB = torch.cat(real_A,fake_B_new,2)
    else
        fake_AB = fake_B_new -- unconditional GAN, only penalizes structure in B
    end
end

-- create closure to evaluate f(X) and df/dX of discriminator
local fDx = function(x)
    --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --Rev_netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)

    gradParametersD:zero()
    
    -- Real
    local output = netD:forward(real_AB)  ----descriminator output of clean input
    --- label smoothening for better prediction in discriminator
    local rand_num=0.7+0.5*torch.rand(1):float()
   -- local label = torch.FloatTensor(output:size()):fill(real_label)
    local label = rand_num:repeatTensor(output:size())
    if opt.gpu>0 then 
    	label = label:cuda()
    end
    
    local errD_real = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(real_AB, df_do:mul(opt.lambdagan))
    
    -- Fake
    local output = netD:forward(fake_AB) ---descriminator output for generated output
    local rand_num2=0.3*torch.rand(1):float()
    --label:fill(fake_label)--*rand_num2:float()----------------------------------------------
    label=rand_num2:repeatTensor(output:size()):cuda()
    local errD_fake = criterion:forward(output, label)
    local df_do = criterion:backward(output, label)
    netD:backward(fake_AB, df_do:mul(opt.lambdagan))
    
    errD = (errD_real + errD_fake)/2
    
    return errD, gradParametersD
end

-- create closure to evaluate f(X) and df/dX of generator
local fGx = function(x)
   -- netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --Rev_netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    gradParametersG:zero()

    
    -- GAN loss
    local df_dg = torch.zeros(fake_AB:size())
    if opt.gpu>0 then 
    	df_dg = df_dg:cuda();
    end
    
    -- gan update to generator from discriminator
    if opt.use_GAN==1 then
       local output = netD.output -- netD:forward{input_A,input_B} was already executed in fDx, so save computation, netD o/p for generated data
       local rand_num1=0.7+0.5*torch.rand(1):float()
       --local label = torch.FloatTensor(output:size()):fill(real_label) -- fake labels are real for generator cost
       local label = rand_num1:repeatTensor(output:size())
       if opt.gpu>0 then 
       	label = label:cuda();
       	end
       errG = criterion:forward(output, label)
       local df_do = criterion:backward(output, label)
       df_dg = netD:updateGradInput(fake_AB, df_do):narrow(2,fake_AB:size(2)-output_nc+1, output_nc)

    else
        errG = 0
    end
    
    -- gradient module error backpropogation to generator
    local df_do_A = torch.zeros(fake_B:size())
    --df_do_A_new=torch.zeros(fake_B:size())
    if opt.gpu>0 then 
    	df_do_A = df_do_A:cuda();
        --df_do_A_new=df_do_A_new:cuda()
    end
    if opt.use_L1==1 then
       errL = criterionA:forward(fake_B, real_B)     
       df_do_A = criterionA:backward(fake_B, real_B)        
       df_do_A_new=model:updateGradInput(fake_B,df_do_A)  -------------- gradient of grad loss wrt genrated ip

       errL1 = criterionA1:forward(fake_B1, real_B1)
       df_do_A1 = criterionA1:backward(fake_B1, real_B1)     
       df_do_A1_new=model1:updateGradInput(fake_B,df_do_A1)  --------------

        errL2 = criterionA2:forward(fake_B2, real_B2)
        df_do_A2 = criterionA2:backward(fake_B2, real_B2)
        df_do_A2_new=model2:updateGradInput(fake_B,df_do_A2)  --------------

       errL3= criterionA3:forward(fake_B3, real_B3)   
       df_do_A3 = criterionA3:backward(fake_B3, real_B3)   
       df_do_A3_new=model3:updateGradInput(fake_B,df_do_A3)  --------------

       errL4 = criterionA4:forward(fake_B4, real_B4)       
       df_do_A4 = criterionA4:backward(fake_B4, real_B4)
       df_do_A4_new=model4:updateGradInput(fake_B,df_do_A4)  --------------
       
       df_do_AE_new=df_do_A4_new:mul(1)+df_do_A3_new:mul(.1)+df_do_A2_new:mul(0.01)+df_do_A1_new:mul(0.001)+df_do_A_new:mul(0.0001)
    else
        errL1 = 0
    end

    --reblur module updation to generator
    errRevG=Rev_criterion:forward(Rev_fake_B,real_A)
    Rev_df_do_new = Rev_criterion:backward(Rev_fake_B, real_A)
    Rev_df_do=Rev_netG:updateGradInput(fake_B_new,Rev_df_do_new)
    --update generator based on all three modules (gan, grad and reblur)
    netG:backward(real_A, df_dg:mul(opt.lambdagan) + df_do_AE_new:mul(opt.lambdagrad)+Rev_df_do:mul(opt.lambdacnn)) -- df_do_AE:mul(0)
    
    return errG, gradParametersG
end

local fRevGx = function(x)
    --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --Rev_netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    
    
    gradParametersRevG:zero()
     --reblur module updation 
    Rev_netG:backward(fake_B_new,Rev_df_do_new)
    
    return errRevG, gradParametersRevG
end



-- train
local best_err = nil
paths.mkdir(opt.checkpoints_dir)
paths.mkdir(opt.checkpoints_dir .. '/' .. opt.name)

-- save opt
file = torch.DiskFile(paths.concat(opt.checkpoints_dir, opt.name, 'opt.txt'), 'w')
file:writeObject(opt)
file:close()

-- parse diplay_plot string into table
opt.display_plot = string.split(string.gsub(opt.display_plot, "%s+", ""), ",")
for k, v in ipairs(opt.display_plot) do
    if not util.containsValue({"errG", "errD", "errL1"}, v) then 
        error(string.format('bad display_plot value "%s"', v)) 
    end
end

-- display plot config
local plot_config = {
  title = "Loss over time",
  labels = {"epoch", unpack(opt.display_plot)},
  ylabel = "loss",
}

-- display plot vars
local plot_data = {}
local plot_win

local counter = 0
for epoch = 1, opt.niter do
    epoch_tm:reset()
    for i = 1, math.min(data:size(), opt.ntrain), opt.batchSize do
        tm:reset()
        
        -- load a batch and run G on that batch
        createRealFake()
        
        -- (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        if opt.use_GAN==1 then optim.adam(fDx, parametersD, optimStateD) end
        
        -- (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
              
        optim.adam(fGx, parametersG, optimStateG)
        optim.adam(fRevGx, parametersRevG, optimStateRevG)

        -- display
        counter = counter + 1
        if counter % opt.display_freq == 0 and opt.display then
            createRealFake()
            if opt.preprocess == 'colorization' then 
                local real_A_s = util.scaleBatch(real_A:float(),100,100)
                local fake_B_s = util.scaleBatch(fake_B:float(),100,100)
                local real_B_s = util.scaleBatch(real_B:float(),100,100)
                disp.image(util.deprocessL_batch(real_A_s), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocessLAB_batch(real_A_s, fake_B_s), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocessLAB_batch(real_A_s, real_B_s), {win=opt.display_id+2, title=opt.name .. ' target'})
            else
                disp.image(util.deprocess_batch(util.scaleBatch(real_A:float(),100,100)), {win=opt.display_id, title=opt.name .. ' input'})
                disp.image(util.deprocess_batch(util.scaleBatch(fake_B_new:float(),100,100)), {win=opt.display_id+1, title=opt.name .. ' output'})
                disp.image(util.deprocess_batch(util.scaleBatch(real_B_new:float(),100,100)), {win=opt.display_id+2, title=opt.name .. ' target'})
            end
        end
      
        -- write display visualization to disk
        --  runs on the first batchSize images in the opt.phase set
        if counter % opt.save_display_freq == 0 and opt.display then
            local serial_batches=opt.serial_batches
            opt.serial_batches=1
            opt.serial_batch_iter=1
            
            local image_out = nil
            local N_save_display = 10 
            local N_save_iter = torch.max(torch.Tensor({1, torch.floor(N_save_display/opt.batchSize)}))
            for i3=1, N_save_iter do
            
                createRealFake()
                print('save to the disk')
                if opt.preprocess == 'colorization' then 
                    for i2=1, fake_B:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0
                        else image_out = torch.cat(image_out, torch.cat(util.deprocessL(real_A[i2]:float()),util.deprocessLAB(real_A[i2]:float(), fake_B[i2]:float()),3)/255.0, 2) end
                    end
                else
                    for i2=1, fake_B_new:size(1) do
                        if image_out==nil then image_out = torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B_new[i2]:float()),3)
                        else image_out = torch.cat(image_out, torch.cat(util.deprocess(real_A[i2]:float()),util.deprocess(fake_B_new[i2]:float()),3), 2) end
                    end
                end
            end
            image.save(paths.concat(opt.checkpoints_dir,  opt.name , counter .. '_train_res.png'), image_out)
            
            opt.serial_batches=serial_batches
        end
        
        -- logging and display plot
        if counter % opt.print_freq == 0 then
            local loss = {errG=errG and errG or -1, errD=errD and errD or -1, errL1=errL1 and errL1 or -1,errRevG=errRevG and errRevG or -1}
            local curItInBatch = ((i-1) / opt.batchSize)
            local totalItInBatch = math.floor(math.min(data:size(), opt.ntrain) / opt.batchSize)
            print(('Epoch: [%d][%1d / %1d]\t  '
                    .. '  Err_G: %.4f Err_RevG: %.4f Err_D: %.4f  ErrL: %.4f ErrL4: %.4f'):format(
                     epoch, curItInBatch, totalItInBatch,              
                     errG, errRevG,errD, errL, errL4))
           
            local plot_vals = { epoch + curItInBatch / totalItInBatch }
            for k, v in ipairs(opt.display_plot) do
              if loss[v] ~= nil then
               plot_vals[#plot_vals + 1] = loss[v] 
             end
            end

            -- update display plot
            if opt.display then
              table.insert(plot_data, plot_vals)
              plot_config.win = plot_win
              plot_win = disp.plot(plot_data, plot_config)
            end
        end
        
        -- save latest model
        if counter % opt.save_latest_freq == 0 then
            print(('saving the latest model (epoch %d, iters %d)'):format(epoch, counter))
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_G.t7'), netG:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_D.t7'), netD:clearState())
            torch.save(paths.concat(opt.checkpoints_dir, opt.name, 'latest_net_RevG.t7'), Rev_netG:clearState())
        end
        
    end
-----------------------------------------Updating Learning Rate---------------------------------
------------------------------------------------------------------------------------------------
   if epoch > opt.niter then
      UpdateLearningRate(opt)
   end
    


--------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------
    
    parametersD, gradParametersD = nil, nil -- nil them to avoid spiking memory
    parametersG, gradParametersG = nil, nil
    parametersRevG, gradParametersRevG = nil, nil

    if epoch % opt.save_epoch_freq == 0 then
        torch.save(paths.concat(opt.checkpoints_dir, opt.name,  epoch .. '_net_G.t7'), netG:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_D.t7'), netD:clearState())
        torch.save(paths.concat(opt.checkpoints_dir, opt.name, epoch .. '_net_RevG.t7'), Rev_netG:clearState())
    end
    
    print(('End of epoch %d / %d \t Time Taken: %.3f'):format(
            epoch, opt.niter, epoch_tm:time().real))
    parametersD, gradParametersD = netD:getParameters() -- reflatten the params and get them
    parametersG, gradParametersG = netG:getParameters()
    parametersRevG, gradParametersRevG =Rev_netG:getParameters()
end
