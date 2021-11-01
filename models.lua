require 'nngraph'

function defineG_encoder_decoder(input_nc, output_nc, ngf)
    local netG = nil 
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})

    return netG
end

function defineG_unet(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineG_unet_128(input_nc, output_nc, ngf)
    -- Two layer less than the default unet to handle 64x64input
    local netG = nil
    -- input is (nc) x 128X128
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 64X64
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 32X32
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 16x 16
     local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)

      local d3_ = e4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 , ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4) - nn.Dropout(0.5)
    -- input is (ngf * 4) x 8 x 8
    local d3 = {d3_,e3} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4*2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 16 x 16
    local d4 = {d4_,e2} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf , 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf ) x 32 x 32
    local d5 = {d5_,e1} - nn.JoinTable(2)

   local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf*2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x128 x 128
    
    local o1 = d6 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineD_basic(input_nc, output_nc, ndf)
    n_layers = 2
    return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

-- rf=1
function defineD_pixelGAN(input_nc, output_nc, ndf)
    local netD = nn.Sequential()
    
    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    netD:add(nn.Sigmoid())
    -- state size: 1 x 256 x 256
        
    return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
    if n_layers==0 then
        return defineD_pixelGAN(input_nc, output_nc, ndf)
    else
    
        local netD = nn.Sequential()
        
         netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
        local nf_mult = 1
        local nf_mult_prev = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        end
        
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
        -- state size: 1 x (N-2) x (N-2)
        
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
    end
end









function defineD_n_layers_text(input_nc, output_nc, ndf, n_layers)
    if n_layers==0 then
        return defineD_pixelGAN(input_nc, output_nc, ndf)
    else
    
        local netD = nn.Sequential()
        
         netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
      
        for n = 1, 3 do 
            netD:add(nn.SpatialConvolution(ndf , ndf , 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
        end
        
         netD:add(nn.SpatialConvolution(ndf , ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf)):add(nn.LeakyReLU(0.2, true))
        netD:add(nn.SpatialConvolution(ndf , ndf, 4, 4, 2, 2, 1, 1))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.View(4*4*ndf))
        -- state size: 1 x (N-2) x (N-2)
        netD:add(nn.Linear(4*4*ndf,1))
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
    end
end
require 'nn'


function create_cnn_10(nc,nz,ngf)
	local netG = nn.Sequential()
	netG = nn.Sequential()
	netG:add(nn.SpatialConvolution(nc, ngf, 5, 5,1,1,2,2))
	netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, 2*ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(2*ngf))
	netG:add(nn.ReLU(true))
	netG:add(nn.SpatialConvolution(2*ngf, 4*ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(4*ngf))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(4*ngf, 8*ngf, 5, 5, 1, 1, 2, 2))
        netG:add(nn.Dropout(0.2))
	netG:add(nn.SpatialBatchNormalization(8*ngf))
	netG:add(nn.ReLU(true))
	netG:add(nn.SpatialConvolution(8*ngf, 16*ngf, 5, 5, 1, 1, 2, 2))
        netG:add(nn.Dropout(0.2))
	netG:add(nn.SpatialBatchNormalization(16*ngf ))
	netG:add(nn.ReLU(true))
	netG:add(nn.SpatialConvolution(16*ngf , 8*ngf, 5, 5, 1, 1,2,2))
	netG:add(nn.SpatialBatchNormalization(8*ngf))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(8*ngf, 4*ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(4*ngf))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(4*ngf, 2*ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(2*ngf))
	netG:add(nn.ReLU(true))        
	netG:add(nn.SpatialConvolution(2*ngf, ngf, 5, 5, 1, 1, 2,2))
        netG:add(nn.SpatialBatchNormalization(ngf))
	netG:add(nn.ReLU(true))   
        netG:add(nn.SpatialConvolution(ngf, nc, 5, 5, 1, 1, 2,2))
	netG:add(nn.Tanh())
return netG
end





function create_cnn(nc,nz,ngf)
	local netG = nn.Sequential()
	netG = nn.Sequential()
	netG:add(nn.SpatialConvolution(nc, ngf, 7, 7, 1, 1, 3, 3))
        netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, ngf, 7, 7, 1, 1, 3, 3))
        netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, ngf, 7, 7, 1, 1, 3, 3))
        netG:add(nn.Dropout(0.2))
        netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))   
	
	netG:add(nn.SpatialConvolution(ngf , ngf, 7,7, 1, 1,3,3))
        netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, nc, 7, 7, 1, 1, 3, 3))
	
	netG:add(nn.Tanh())
return netG
end



function create_cnn1(nc,nz,ngf)
	local netRevG = nn.Sequential()
	netRevG = nn.Sequential()
	netRevG:add(nn.SpatialConvolution(nc, ngf, 5, 5,1,1,2,2))
        netRevG:add(nn.SpatialBatchNormalization(ngf ))
	netRevG:add(nn.ReLU(true))
 
	netRevG:add(nn.SpatialConvolution(ngf, ngf, 5, 5, 1, 1, 2, 2))
        netRevG:add(nn.SpatialBatchNormalization(ngf ))
	netRevG:add(nn.ReLU(true))
	netRevG:add(nn.SpatialConvolution(ngf, ngf, 5, 5, 1, 1, 2, 2))
        netRevG:add(nn.SpatialBatchNormalization(ngf ))
	netRevG:add(nn.ReLU(true))
	netRevG:add(nn.SpatialConvolution(ngf , ngf, 5, 5, 1, 1,2,2))
        netRevG:add(nn.SpatialBatchNormalization(ngf ))
	netRevG:add(nn.ReLU(true))
        
	netRevG:add(nn.SpatialConvolution(ngf, nc, 5, 5, 1, 1, 2,2))
        netRevG:add(nn.Tanh())
return netRevG
end






function create_cnn_circles(nc,nz,ngf)
	local netG = nn.Sequential()
	netG = nn.Sequential()
	netG:add(nn.SpatialConvolution(nc, ngf, 5, 5,1,1,2,2))
	netG:add(nn.SpatialBatchNormalization(ngf ))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(ngf))
	netG:add(nn.ReLU(true))
	netG:add(nn.SpatialConvolution(ngf, ngf, 5, 5, 1, 1, 2, 2))
	netG:add(nn.SpatialBatchNormalization(ngf))
	netG:add(nn.ReLU(true))
        netG:add(nn.SpatialConvolution(ngf, ngf, 5, 5, 1, 1, 2, 2))
        netG:add(nn.Dropout(0.2))
	netG:add(nn.SpatialBatchNormalization(ngf))
	netG:add(nn.ReLU(true))
	netG:add(nn.SpatialConvolution(ngf, nc, 5, 5, 1, 1, 2, 2))
      	netG:add(nn.Tanh())
return netG
end


