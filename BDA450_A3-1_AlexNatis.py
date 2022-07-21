# Base for simple simulation of virus transmission between pedestrians on a sidewalk.
# Written by Reid Kerr for use in an assignment in BDA450.
# Please note that this code is not written to be efficient/elegant/etc.!  It is written
# to be brief/simple/transparent, to a reasonable degree.
# There may very well be bugs present!  Moreover, there is very little validation of inputs
# to function calls, etc.  It is generally up to the developer who uses this to build their simulation
# to ensure that their valid state is maintained.

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.animation import FuncAnimation
import random

rand = random.Random()
SIDEWALK_WIDTH = 10  # This is the y-dimension of the sidewalk
SIDEWALK_LENGTH = 200  # This is the x-dimension of the sidewalk
TRANSPROB = 0.1  # Probability of transmission of virus in 1 time step at distance 1
INFECTED_PROP = 0.1  # The proportion of people who enter the simulation already carrying the virus
INTERARRIVAL = 3 # Average number of time steps between arrivals (each side handled separately)

# Setup for graphical display
colourmap = colors.ListedColormap(["lightgrey", "green", "red", "yellow"])
normalizer = colors.Normalize(vmin=0.0, vmax=3.0)


# An agent representing a single person traversing the sidewalk.  Simple movement is demonstrated.  It is up
# to the user to implement behaviour according to the assignment specification, and to collect data as
# necessary.
# A person occupies an x,y position on the sidewalk.  While the person can access their own position, they
# cannot change their coordinates directly.  Instead, they must make movement requests to the sidewalk, which
# (if the move is valid) updates the person's x and y coordinates.
class Person:
    def __init__(self, id, sidewalk, direction):
        self.id = id
        self.active = False
        self.sidewalk = sidewalk
        self.infected = True if rand.random() < INFECTED_PROP else False
        self.newlyinfected = False
        self.direction = direction
        self.bornInfected = False

        if self.infected == True:
            self.bornInfected = True

        if self.direction == 1:
            self.startx = 0
            self.x = self.startx
        else:
            self.startx = SIDEWALK_LENGTH - 1
            self.x = self.startx

        self.starty = rand.randint(0, SIDEWALK_WIDTH - 1)
        self.y = self.starty

    def enter_sidewalk(self, x, y):
        if self.sidewalk.enter_sidewalk(self, x, y):
            self.active = True

    # This is the method that is called by the simulation once for each time step.  It is during this call
    # that the agent is active and can take action: examining surroundings, attempting to move, etc.
    # This method should only be called when the agent's 'active' flag is true, but you might want to check here
    # as well for safety.
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################
    def step(self):

        if self.direction == 1: # Check which direction we're going
            desiredx = self.x + 1
            desiredy = self.y
        else:
            desiredx = self.x - 1
            desiredy = self.y

        if self.sidewalk.isoccupied(desiredx, desiredy):
            highy = desiredy + 1
            lowy = desiredy -1
            direct = random.uniform(0,1)

            if not self.sidewalk.storage.isoccupied(desiredx, highy) and not self.sidewalk.storage.isoccupied(desiredx, lowy): # if diag left/right both open, 50/50 decide dir
                if direct > 0.5:
                    desiredx = self.x
                    desiredy = highy
                else:
                    desiredx = self.x
                    desiredy = lowy
            elif not self.sidewalk.storage.isoccupied(desiredx, highy): # check if top is available
                desiredx = self.x
                desiredy = highy
            elif not self.sidewalk.storage.isoccupied(desiredx, lowy): # check if bottom is available
                desiredx = self.x
                desiredy = lowy
            else:
                desiredx = self.x
                desiredy = self.y

        # Ensure x and y don't go off edge of sidewalk
        desiredx = max(min(desiredx, SIDEWALK_LENGTH - 1), 0)
        desiredy = max(min(desiredy, SIDEWALK_WIDTH - 1), 0)

        self.sidewalk.attemptmove(self, desiredx, desiredy)

        
################################################################################################################################################################
################################################################################################################################################################
################################################################################################################################################################

    def __str__(self):
        return "id: %d  x: %d  y: %d" % (self.id, self.x, self.y)


# The class representing the sidewalk itself.  Agents must enter the sidewalk to be active in the simulation.
# The sidewalk controls agents' positions/movement.
class Sidewalk:
    arrL = 1
    arrR = 1

    def __init__(self):
        # Tracking of positions of agents
        self.storage = SWGrid()


        # Bitmap is for graphical display
        self.bitmap = [[0.0 for i in range(SIDEWALK_LENGTH)] for j in range(SIDEWALK_WIDTH)]

    # An agent must enter the sidewalk at one of the ends (i.e., with an x coordinate of either zero or
    # the maximum.  They may attempt to enter at any y coordinate.  The function returns true if successful, false
    # if the agent is not added to the sidewalk (e.g., if the desired square is already occupied.)
    # The method will set the agent's x and y position if the attempt is successful.
    def enter_sidewalk(self, person, x, y):
        # New entrant to the sidewalk, must attempt to start at one end
        if x!=0 and x!=SIDEWALK_LENGTH-1:
            print("Must start at an end!")
            return False

        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            print("Move rejected: occupied")
            return False
        self.storage.add_item(x, y, person)
        person.x = x
        person.y = y
        return True

    # An agent must leave the sidewalk at one of the ends (i.e., with an x coordinate of either zero or
    # the maximum.  The function returns true if successful, false if not.
    # You should be sure to get any information you want from the agent before doing so, because you may not
    # have a handle for it afterwards.
    def leave_sidewalk(self, person):
        # Must attempt to leave at one end
        if person.x != 0 and person.x != SIDEWALK_LENGTH - 1:
            print("Must leave at an end!")
            return False

        self.storage.remove_item(person)

    # Returns True if person successfully moved, False if not (e.g., the desired square is occupied).
    # An agent can only move one square in a cardinal direction from its current position; any other attempt will
    # be rejected.
    # The method will set the agent's x and y position if the attempt is successful.
    def attemptmove(self, person, x, y):

        # Reject move of more than 1 square
        if (abs(person.x - x) + abs(person.y - y)) > 1:
            print("Attempt to move more than one square!")
            return False

        # Only allow move if space not currently occupied
        if self.storage.isoccupied(x, y):
            # print("Move rejected: occupied")
            return False
        person.x = x
        person.y = y
        self.storage.move_item(x, y, person)
        return True

    # When called, infects new agents (with some probability) who are in proximity to infected agents.  The risk
    # is equal to the simulation parameter at distance of 1, and decreases with greater distance.
    # You may add to this function, e.g., for gathering data, but do not modify the actual determination of infection.
    def spread_infection(self):
        newInfected = 0
        for person in self.storage.get_list():
            currentx = person.x
            currenty = person.y
            if person.infected:
                # Find all agents within a square of 'radius' 2 of the infected agent
                for x in range(currentx - 2, currentx + 3):
                    for y in range(currenty - 2, currenty + 3):
                        target = self.storage.get_item(x, y)

                        # If target is not infected, infect with probability dependent on distance
                        if target is not None and not target.infected: #############################################  and currentx != 0 and currenty != 0
                            riskfactor = 1 / ((currentx - x) ** 2 + (currenty - y) ** 2)
                            tranmission_prob = TRANSPROB * riskfactor
                            if rand.random() < tranmission_prob:
                                target.infected = True
                                target.newlyinfected = True
                                newInfected += 1
                                # print('New infection! %s' % target)
        return newInfected

    # Updates the graphic for display
    def refresh_image(self):
        self.bitmap = [[0.0 for i in range(SIDEWALK_LENGTH)] for j in range(SIDEWALK_WIDTH)]
        for person in self.storage.get_list():
            x = person.x
            y = person.y
            colour = 1
            if person.newlyinfected:
                colour = 3
            elif person.infected:
                colour = 2
            self.bitmap[y][x] = colour

    # Function that is called at each time step, to execute the step.  Calls the step() function of every active
    # agent, spreads infection after the agents have moved, and updates the image for display.  You will need to add
    # code here to, for example, have new agents enter.
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################
    def run_step(self, time_step, id_count, results, totalInfected, bornInfectedO, newInfectedO, nonInfectedO):
        results[t] = []
        if time_step >= self.arrL:
            id_count += 1
            person = Person(id_count, self, 1)
            person.enter_sidewalk(person.startx, person.starty)
            self.storage.add_item(person.x, person.y, person)
            self.arrL = int(random.expovariate(1/INTERARRIVAL)) + time_step

        if time_step >= self.arrR:
            id_count += 1
            person = Person(id_count, self, -1)
            person.enter_sidewalk(person.startx, person.starty)
            self.storage.add_item(person.x, person.y, person)
            self.arrR = int(random.expovariate(1/INTERARRIVAL)) + time_step

        bornInfected = 0
        currentInfected = 0
        newlyInfected = 0
        newInfections = 0
        nonInfected = 0
        for person in self.storage.get_list():
            if person.active:
                if person.infected == True:
                    currentInfected += 1
                if person.bornInfected == True:
                    bornInfected += 1
                if person.newlyinfected == True:
                    newlyInfected += 1
                else:
                    nonInfected += 1
                person.step()
                tooFar = person.x + 1
                tooShort = person.x - 1
                if tooFar > SIDEWALK_LENGTH-1 or tooShort < 0:
                    if person.infected == True:
                        totalInfected += 1
                    if person.bornInfected == True:
                        bornInfectedO += 1
                    if person.newlyinfected == True:
                        newInfectedO += 1
                    if person.infected == False:
                        nonInfectedO += 1
                    self.leave_sidewalk(person)

        newInfections += self.spread_infection()
        self.refresh_image()
        results[t].append(id_count)
        results[t].append(len(self.storage.get_list()))
        results[t].append(currentInfected)
        results[t].append(newlyInfected)
        results[t].append(nonInfected)
        results[t].append(newInfections)
        results[t].append(bornInfected)
        results[t].append(totalInfected + currentInfected)
        results[t].append(bornInfectedO + bornInfected)
        results[t].append(newInfectedO + newlyInfected)
        results[t].append(nonInfectedO + nonInfected)

        return id_count, results, totalInfected, bornInfectedO, newInfectedO, nonInfectedO
##########################################################################################################################################################
##########################################################################################################################################################
##########################################################################################################################################################

    # Returns true if x,y is occupied by an agent, false otherwise.  This is the only information that an agent
    # has about other agents; it can't (e.g.) see if other agents are infected!
    def isoccupied(self, x, y):
        return self.storage.isoccupied(x, y)


# Used to provide storage, lookup of occupants of sidewalk
class SWGrid:
    def __init__(self):
        self.dic = dict()

    def isoccupied(self, x, y):
        # self.check_coordinates(x, y)
        return (x, y) in self.dic

    # Stores item at coordinates x, y.  Throws an exception if the coordinates are invalid.  Returns false if
    # unsuccessful (e.g., the square is occupied) or true if successful.
    def add_item(self, x, y, item):
        self.check_coordinates(x, y)
        if (x, y) in self.dic:
            return False
        self.dic[(x, y)] = item
        return True

    # Removes item from its current coordinates (which do not need to be provided) and stores it
    # at coordinates x, y.  Throws an exception if the coordinates are invalid or if the square is occupied.
    def move_item(self, x, y, item):
        self.check_coordinates(x, y)
        if self.isoccupied(x, y):
            raise Exception("Move to occupied square!")

        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        del self.dic[oldloc]
        self.add_item(x, y, item)

    # Removes item (coordinates do not need to be provided)
    # Throws an exception if the item doesn't exist.
    def remove_item(self, item):
        # Find and remove previous location.  Assumed state is valid (meaning only one entry per x,y key)
        oldloc = next(key for key, value in self.dic.items() if value == item)
        if oldloc is None:
            raise Exception('Attempt to remove non-existent item!')
        del self.dic[oldloc]

    def get_item(self, x, y):
        # self.check_coordinates(x, y)
        return self.dic.get((x, y), None)

    # Returns a list of all agents in the simulation.
    def get_list(self):
        return list(self.dic.values())

    def check_coordinates(self, x, y):
        if x < 0 or x >= SIDEWALK_LENGTH or y < 0 or y >= SIDEWALK_WIDTH:
            raise Exception('Illegal Coordinate')


#
# Run simulation
#

sw = Sidewalk()

# This is NOT how people should be added-- it is just to put agents in place to demonstrate random movement.
# Agents should be added by a random process in the run_step method.
# personlist = [Person(i, sw) for i in range(40)]
# for person in personlist:
#     person.enter_sidewalk(person.startx, person.starty)

# Set up graphical display
display = plt.figure(figsize=(15, 5))
image = plt.imshow(sw.bitmap, cmap=colourmap, norm=normalizer, animated=True)

# External Tracking vars
t = 0
id_count = 0
totalInfected = 0
bornInfectedO = 0
newInfectedO = 0
nonInfectedO = 0
results = {}
# The graphical routine runs the simulation 'clock'; it calls this function at each time step.  This function
# calls the sidewalk's run_step function, as well as updating the display.  You should not implement your simulation
# here, but instead should do so in the run_step method.
def updatefigure(*args):
    global t
    global id_count
    global results
    global totalInfected
    global bornInfectedO
    global newInfectedO
    global nonInfectedO

    t += 1

    if t % 100 == 0:
        print("Time: %d" % t)
    id_count, results, totalInfected, bornInfectedO, newInfectedO, nonInfectedO = sw.run_step(t, id_count, results, totalInfected, bornInfectedO, newInfectedO, nonInfectedO)
    sw.refresh_image()
    image.set_array(sw.bitmap)
    return image,


# Sets up the animation, and begins the process of running the simulation.  As configured below, it will
# run for 1000 steps.  After this point, it will simply stop, but the window will remain open.  You can close
# the window to proceed to the code below these lines (where you could add, for example, output of your statistics.
#
# You can change the speed of the simulation by changing the interval, and the duration by changing frames.
anim = FuncAnimation(display, updatefigure, frames=350, interval=100, blit=True, repeat=False)
plt.show()
x = np.arange(len(results))
finalResults = pd.DataFrame.from_dict(results, orient='index',
                                          columns=['Total_Agents', 'Num_Active_Agents', 'Current_Infected', 'Num_Newly_Infected', 
                                                   'Num_Non_Infected', 'New_Infections', 'Num_Born_Infected', 'Total_Infected', 
                                                   'Born_Infected_Overall', 'Newly_Infected_Overall', 'Non_Infected_Overall'])

print(finalResults.tail(5))
plt.style.use('ggplot')
plt.title("Current Vars vs. Time")
plt.ylabel("Count/Number")
plt.xlabel("Time (s)")
plt.plot(finalResults['Num_Active_Agents'], label='Num Active Agents', color='blue', lw=2, alpha=1)
plt.plot(finalResults['Current_Infected'], label='Total Infected', color='orange', lw=2, alpha=0.5)
plt.plot(finalResults['Num_Non_Infected'], label='Num Non Infected', color='green', lw=2, alpha=1)
plt.plot(finalResults['Num_Newly_Infected'], label='Num Newly Infected', color='yellow', lw=2, alpha=1)
plt.plot(finalResults['Num_Born_Infected'], label='Num Born Infected', color='red', lw=2, alpha=1)
plt.legend(loc='best')
plt.show()

plt.title("Overall Vars vs. Time")
plt.ylabel("Count/Number")
plt.xlabel("Time (s)")
plt.plot(finalResults['Total_Agents'], label='Total Agents', color='black', lw=2, alpha=1)
plt.plot(finalResults['Total_Infected'], label='Total Infected', color='red', lw=2, alpha=1)
plt.plot(finalResults['Non_Infected_Overall'], label='Non Infected Overall', color='green', lw=2, alpha=1)
plt.plot(finalResults['Newly_Infected_Overall'], label='Newly Infected Overall', color='yellow', lw=2, alpha=1)
plt.plot(finalResults['Born_Infected_Overall'], label='Born Infected Overall', color='red', lw=2, alpha=1)
plt.legend(loc='best')
plt.show()

initialInfectionRate = ((finalResults.iloc[-1,8]) / (finalResults.iloc[-1,0]))
newInfectionRate = ((finalResults.iloc[-1,9]) / (finalResults.iloc[-1,0]))
totalInfectionRate = ((finalResults.iloc[-1,7]) / (finalResults.iloc[-1,0]))

print(f'Initial Infection Rate: {initialInfectionRate:.3f}')
print(f'New Infection Rate: {newInfectionRate:.3f}')
print(f'Total Infection Rate: {totalInfectionRate:.3f}')

print("Done!")

##########################################################