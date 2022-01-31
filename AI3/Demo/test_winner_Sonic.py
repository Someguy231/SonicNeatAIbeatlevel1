import retro
import numpy as np
import cv2 
import neat
import pickle

env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')

imgarray = []

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-sonic')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('Sonic-winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.FeedForwardNetwork.create(genome, config)

score_current=0
max_score_current=0
fitness = 0
ring_current = 0
max_ring=0
fitness = 0
counter = 0
frame=0
lives=0
done=False
end_level=0
while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    for x in ob:
        for y in x:
            imgarray.append(y)
    
    action = net.activate(imgarray)
    
    ob, rew, done, info = env.step(action)
    imgarray.clear()
    
    lives=info["lives"]
    ring_current = info['rings']
    score_current+=rew
    lives=info['lives']
    end_level=info['level_end_bonus']
    
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
        fitness+=counter
                
        print(fitness)
