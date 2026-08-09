import pandas as pd
from typing import NamedTuple
from logger.logging_module import log, initilaize_logger, lte
import random, torch
import numpy as np

# TODO - to introduce randomness? storing 2 highest loss sequences, and quering randomly from them 
# UPDATE proportion between baseline loss and current batch loss is source of selection between random and bufor, probalby storing 2 loses to select is not neccesary

class CBPayload(NamedTuple):
    """
    Payload for curiosity buffor
    """
    start_state : int
    seq_loss    : float
    end_state   : int
    action_seq  : torch.tensor


class CuriosityBuffor:

    def __init__(self, max_size, loss_baseline):

        self.max_size = max_size
        self.loss_baseline = loss_baseline
        self.intialize_table()


    def insert_df(self, payload : CBPayload):
         """
         Inserting or updating seuqences in our buffor in order to keep buffor usefull
         NONE       - new seuqence did not qualify for buffor 
         NEW        - new sequence for particualr state has been inserted that did not exist before
         LOSS INC   - we have found seuqence with higher loss, so we update our buffor for that particular state
         LOSS DEC   - our model that is source of loss value for sequence has learned to predict sequence better, so we decreae
                      loss in buffor accordingly
         """
         status = "NONE"
         if (payload.end_state < self.max_size) and (payload.start_state != payload.end_state):
            if (payload.start_state not in self.buffor.index):
                status = "NEW"
                self.buffor.loc[payload.start_state] = {
                    "action_sequence"  : payload.action_seq, 
                    "end_state"        : payload.end_state, 
                    "loss_value"       : payload.seq_loss
                }
            else:
                if not self.check_for_update(payload):
                    if self.buffor.loc[payload.start_state, "loss_value"] < payload.seq_loss:
                        self.buffor.loc[payload.start_state, "loss_value"] = payload.seq_loss
                        self.buffor.at[payload.start_state, "action_sequence"] = payload.action_seq
                        self.buffor.loc[payload.start_state, "end_state"] = payload.end_state
                        status = "LOSS INC"
                else:
                    status = "LOSS DEC"

         log(f"Transition: {payload.start_state}->{payload.end_state},\t loss: {payload.seq_loss:.2f},\t actions: {payload.action_seq} [{status}]")


    def check_for_update(self, payload : CBPayload):

        if (payload.end_state == self.buffor.loc[payload.start_state, "end_state"] and
            payload.seq_loss < self.buffor.loc[payload.start_state, "loss_value"]):
                self.buffor.loc[payload.start_state, "loss_value"] = payload.seq_loss
                self.buffor.at[payload.start_state, "action_sequence"] = payload.action_seq
                return True



    def select_actions(self, start_state):
        """
        It can return random seuqence of actions or take it from buffor. If we have an entry for particular state
        we calculate probability by dividing loss for this sequence by baseline loss (which is loss if our model is fully random)
        If we don't have well modeled those state sequences it means we will have 100% probability of taknig sequence from buffor
        If our loss is low (we predict those correctly) we will have low probability to take from buffor we get random sequence instead
        to look for new action-state sequences
        """
        if start_state in self.buffor.index:
            prob = self.buffor.loc[start_state, "loss_value"] / self.loss_baseline
            roll = random.random()
            if roll < prob:
                log(f"Selecting buffor actions", lte.info)
                return self.buffor.loc[start_state, "action_sequence"]
  
        log(f"Selecting random actions", lte.info)
        return torch.from_numpy(np.random.randint(0, 4, (1, 4))).to(torch.long)[0]
        # TODO - function to sleect not fully random random actions to have certain directioncd ..


    def intialize_table(self):
        """
        Initialization of buffor. It is pandas df, where index is start_state, and we have 3 columns named approprietly
        """
        cols = ["action_sequence", "end_state", "loss_value"]
        self.buffor = pd.DataFrame(columns=cols)
        
