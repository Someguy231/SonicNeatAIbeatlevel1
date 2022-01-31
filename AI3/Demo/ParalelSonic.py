import retro        
import numpy as np  
import cv2          
import neat         
import pickle       


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        #self.env = retro.make('SonicTheHedgehog2-Genesis', 'EmeraldHillZone.Act1')

        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        score_current=0
        max_score_current=0
        fitness = 0
        ring_current = 0
        max_ring=0
        #xpos=0
        #xpos_max=0
        counter = 0
        imgarray = []
        lives=0
        end_level=False
        while not done:
            self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            actions = net.activate(imgarray)
            
            ob, rew, done, info = self.env.step(actions)
            #xpos=info['x']
            ring_current = info['rings']
            score_current+=rew
            lives=info['lives']
            end_level=info['level_end_bonus']
            # if xpos > xpos_max:
            #     fitness += 1
            #     xpos_max = xpos
            if ring_current > max_ring:
                max_ring = ring_current
                counter = 0
                
            else:
                counter += 1
            if max_ring==99:
                max_ring=0
            if score_current > max_score_current:
                max_score_current = score_current
                counter = 0
                
            else:
                counter += 1
            if end_level==True:
                counter=0
                  
            if counter > 1550 or lives<3:
                done = True
                
        fitness+=score_current
                
        print(fitness)
        return fitness

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-sonic')

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint("sonic1-486")
#p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))
if __name__=='__main__':
    
    pe = neat.ParallelEvaluator(10, eval_genomes)
    #This causes parallelization with the number of threads specified with the class eval_genomes
    winner = p.run(pe.evaluate)
    #When you want to save it specify after the evaluate part
    with open('Sonic-winner.pkl', 'wb') as output:
        print('writing winner gen to ', output)
        #This is to signal that it has been recorded
        pickle.dump(winner, output, 1)
        #this dumps only the best 1 of the generation
