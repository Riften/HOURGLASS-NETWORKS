require 'optim'
require 'xlua'

paths.dofile('hg.lua')
paths.dofile('eval.lua')

hg = createModel()
datas = torch.load('data_file/train/data.t7')
datas = datas:float()
print('datas loading done...')
labels = torch.load('data_file/train/label.t7')
labels = labels:float()
print('labels loading done...')
optfn = optim['rmsprop']
batch_size = 8
nIters = 20
nEpochs = 200
optimState = {
    learningRate = 2.5e-4,
    learningRateDecay = 0.0,
    momentum = 0.0,
    weightDecay = 0.0,
    alpha = 0.99,
    epsilon = 1e-8
    }

function applyFn(fn, t, t2)
    -- Apply an operation whether passed a table or tensor
    local t_ = {}
    if type(t) == "table" then
        if t2 then
            for i = 1,#t do t_[i] = applyFn(fn, t[i], t2[i]) end
        else
            for i = 1,#t do t_[i] = applyFn(fn, t[i]) end
        end
    else t_ = fn(t, t2) end
    return t_
end

function accuracy(output,label)
    if type(output) == 'table' then
        return heatmapAccuracy(output[#output],label[#output])
    else
        return heatmapAccuracy(output,label)
    end
end

criterion = nn.MSECriterion()--nn['MSECriterion']()
local function evalFn(x) return criterion.output, gradparam end
cudnn.fastest = true



hg:training()

hg = hg:cuda()
criterion = criterion:cuda()

for i=1,nIters do
	print("==> Starting epoch: " .. i .. "/" .. nIters)
	local avgLoss, avgAcc = 0.0, 0.0
	for epoch=1,nEpochs do
		xlua.progress(epoch,nEpochs)
		local data = datas[{{(epoch-1)*batch_size+1,epoch*batch_size}}]
		local label = labels[{{(epoch-1)*batch_size+1,epoch*batch_size}}]
		data = applyFn(function (x) return x:cuda() end, data)
        label = applyFn(function (x) return x:cuda() end, label)

		local output = hg:forward(data)
		local err = criterion:forward(output, label)
		avgLoss = avgLoss + err / nEpochs

		hg:zeroGradParameters()
		hg:backward(data, criterion:backward(output, label))
		param, gradparam = hg:getParameters()
		optfn(evalFn, param, optimState)

		avgAcc = avgAcc + accuracy(output, label) / nEpochs
	end
	print('Loss   '..avgLoss)
	print('Acc    '..avgAcc)

end
torch.save("trained_model.t7",hg)


--data = applyFn(function (x) return x:cuda() end, img_part)
--labels = applyFn(function (x) return x:cuda() end, label_part)

