require 'grad_model'
function create_grad_model5(nc,no)
local grad=create_grad_model(nc,no)

local model=nn.Sequential()
model:add(grad) 
return model
end

function create_grad_model1(nc,no)
local grad1=create_grad_model(nc,no)

local model1=nn.Sequential()
model1:add(nn.SpatialMaxPooling(2,2,2,2))
model1:add(grad1) 
return model1
end

function create_grad_model2(nc,no)
local grad2=create_grad_model(nc,no)

local model2=nn.Sequential()
model2:add(nn.SpatialMaxPooling(2,2,2,2))
model2:add(nn.SpatialMaxPooling(2,2,2,2))
model2:add(grad2)
return model2
end

function create_grad_model3(nc,no)
local grad3=create_grad_model(nc,no)

local model3=nn.Sequential()
model3:add(nn.SpatialMaxPooling(2,2,2,2))
model3:add(nn.SpatialMaxPooling(2,2,2,2))
model3:add(nn.SpatialMaxPooling(2,2,2,2))
model3:add(grad3)
return model3
end

function create_grad_model4(nc,no)
local grad4=create_grad_model(nc,no)

local model4=nn.Sequential()
model4:add(nn.SpatialMaxPooling(2,2,2,2))
model4:add(nn.SpatialMaxPooling(2,2,2,2))
model4:add(nn.SpatialMaxPooling(2,2,2,2))
model4:add(nn.SpatialMaxPooling(2,2,2,2))
model4:add(grad4)
return model4
end
