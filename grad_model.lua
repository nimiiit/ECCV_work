function create_grad_model(nc,no)

-- Network 1 --------------------------------------
---------------------------------------------------
net1 = nn.Sequential()
net1:add(nn.SpatialConvolution(nc, no, 3, 3,1,1,1,1))
layer=net1:get(1)
layer.bias:zero()
---------------------------------------------------
---------------------------------------------------
t=torch.FloatTensor(3,3):zero()
t[1][1]=0
t[1][2]=1
t[1][3]=0
t[2][1]=1
t[2][2]=-4
t[2][3]=1
t[3][1]=0
t[3][2]=1
t[3][3]=0
w=torch.FloatTensor(3,3,3,3):zero()
w[1][1]=t
w[2][2]=t
w[3][3]=t
layer.weight:zero()
layer.weight:add(w)


-- Network 2 ---------------------------------------
----------------------------------------------------

--[[net2 = nn.Sequential()
net2:add(nn.SpatialConvolution(nc, no, 3, 3,1,1,1,1))
net2:add(nn.Abs())
layer=net2:get(1)
layer.bias:zero()
---------------------------------------------------
---------------------------------------------------
t=torch.FloatTensor(3,3):zero()
t[1][1]=0
t[1][2]=1
t[1][3]=0
t[2][1]=0
t[2][2]=-2
t[2][3]=0
t[3][1]=0
t[3][2]=1
t[3][3]=0

w=torch.FloatTensor(3,3,3,3):zero()
w[1][1]=t
w[2][2]=t
w[3][3]=t
layer.weight:zero()
layer.weight:add(w)

-- Model that concatenates Network 1 and Network 2---
-----------------------------------------------------
parallel_model = nn.ConcatTable()  
parallel_model:add(net1)
parallel_model:add(net2)

-- Putting everything together-----------------------
-----------------------------------------------------
model = nn.Sequential()
model:add(parallel_model)
model:add(nn.CAddTable())

model:add(nn.SpatialConvolution(nc,no,3,3,1,1,1,1))
layer=model:get(3)
layer.bias:zero()

------------------------------------------------------
------------------------------------------------------
t=torch.FloatTensor(3,3):zero()
t[1][1]=1
t[1][2]=1
t[1][3]=1
t[2][1]=1
t[2][2]=1
t[2][3]=1
t[3][1]=1
t[3][2]=1
t[3][3]=1
w=torch.FloatTensor(3,3,3,3):zero()
w[1][1]=t
w[2][2]=t
w[3][3]=t
layer.weight:zero()
layer.weight:add(w)

return model
end

function create_sml_model1(nc,no)

-- Network 1 --------------------------------------
---------------------------------------------------
net11 = nn.Sequential()
net11:add(nn.SpatialConvolution(nc, no, 3, 3,1,1,1,1))
net11:add(nn.Abs())
layer1=net11:get(1)
layer1.bias:zero()
---------------------------------------------------
---------------------------------------------------
t1=torch.FloatTensor(3,3):zero()
t1[1][1]=0
t1[1][2]=0
t1[1][3]=0
t1[2][1]=1
t1[2][2]=-2
t1[2][3]=1
t1[3][1]=0
t1[3][2]=0
t1[3][3]=0
w1=torch.FloatTensor(3,3,3,3):zero()
w1[1][1]=t1
w1[2][2]=t1
w1[3][3]=t1
layer1.weight:zero()
layer1.weight:add(w1)


-- Network 2 ---------------------------------------
----------------------------------------------------

net21 = nn.Sequential()
net21:add(nn.SpatialConvolution(nc, no, 3, 3,1,1,1,1))
net21:add(nn.Abs())
layer1=net2:get(1)
layer1.bias:zero()
---------------------------------------------------
---------------------------------------------------
t1=torch.FloatTensor(3,3):zero()
t1[1][1]=0
t1[1][2]=1
t1[1][3]=0
t1[2][1]=0
t1[2][2]=-2
t1[2][3]=0
t1[3][1]=0
t1[3][2]=1
t1[3][3]=0

w1=torch.FloatTensor(3,3,3,3):zero()
w1[1][1]=t1
w1[2][2]=t1
w1[3][3]=t1
layer1.weight:zero()
layer1.weight:add(w1)

-- Model that concatenates Network 1 and Network 2---
-----------------------------------------------------
parallel_model1 = nn.ConcatTable()  
parallel_model1:add(net11)
parallel_model1:add(net21)

-- Putting everything together-----------------------
-----------------------------------------------------
model1 = nn.Sequential()
model1:add(parallel_model1)
model1:add(nn.CAddTable())

model1:add(nn.SpatialConvolution(nc,no,3,3,1,1,1,1))
layer1=model1:get(3)
layer1.bias:zero()

------------------------------------------------------
------------------------------------------------------
t1=torch.FloatTensor(3,3):zero()
t1[1][1]=1
t1[1][2]=1
t1[1][3]=1
t1[2][1]=1
t1[2][2]=1
t1[2][3]=1
t1[3][1]=1
t1[3][2]=1
t1[3][3]=1
w1=torch.FloatTensor(3,3,3,3):zero()
w1[1][1]=t1
w1[2][2]=t1
w1[3][3]=t1
layer1.weight:zero()
layer1.weight:add(w1)]]

return net1
end
