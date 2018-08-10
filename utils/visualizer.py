import visdom

class Visualizer:
    def __init__(self,env='micro_expression'):
        self.vis=visdom.Visdom(env=env)
    def add_loss(self,x,loss):
        self.vis.line(X=x,Y=loss,win='loss',update='append',opts={'title':'loss','xlabel':'times','ylabel':'loss'})
    def add_accuracy(self,accuracy,epoch):
        self.vis.line(X=epoch,Y=accuracy,win='accuracy',update='append',opts={'title':'accuracy','xlabel':'epoch','ylabel':'accuracy'})
    def add_text1(self,epoch,times,loss):
        self.vis.text('epoch:{}/times:{}/loss{}'.format(epoch,times,loss),win='Text',append=True)
    def add_test2(self,epoch,accuracy):
        self.vis.text('epoch:{}/accuracy:{}'.format(epoch,accuracy),win='Text',append=True)
