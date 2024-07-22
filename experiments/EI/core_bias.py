'''
BiasGen
'''
class CoreBias:
    '''
    Customize neuron modules
    '''
    N_CORES = 4
    def __init__(self):
        self.core_spec = {}

    @staticmethod
    def get_module_params(neuron_type="Pyr"):
        '''
        Set biases for a neuron Type
        '''
        neuron_spec= dict()
        conn_spec = dict()
        weight_spec = dict()
        adaptation_spec = dict()
        NMDA_theta = (0,0)
        mismatch_delay = (1,80)
        if neuron_type=="Pyr": # set for core 2
            neuron_spec = dict(
                        SOIF_LEAK_N = (2,4), # corrosponding to 20 ms time constant (2,5)
                        SOIF_GAIN_N = (3, 40),
                        SOIF_REFR_N = (2,45),#(2, 60),
                        SOIF_DC_P   = (0, 0),
                        SOIF_SPKTHR_P =  (3,40)
                        )
            conn_spec = dict(
                        DEAM_ETAU_P =  (2, 30),#~5-10 msec
                        DEAM_EGAIN_P = (2, 35),# 
                        # DEAM_REV_N = (4,200), # BUG
                        DEGA_ITAU_P =  (1, 50),# GAGA_B, slower and weeker than NMDA 
                        DEGA_IGAIN_P =  (1, 50),
                        # DEGA_REV_P = (4,200),
                        DENM_ETAU_P =  (1, 60),
                        DENM_EGAIN_P =  (1, 130),
                        # DENM_REV_N = (4,200),
                        DENM_NMREV_N = NMDA_theta,
                        DESC_ITAU_P =  (2, 16),# GAGA_A
                        DESC_IGAIN_P =  (1, 90),# GAGA_A
                        SYPD_EXT_N = (3, 200),
                        SYPD_DLY0_P =  (0,0), # Base delay set to 0
                        SYPD_DLY1_P =  (5,250), # Precise delay set to infinity
                        SYPD_DLY2_P =  mismatch_delay  # Mismatch delay
                    )
            adaptation_spec = dict(
                        SOAD_PWTAU_N = (5, 255),
                        SOAD_GAIN_P = (0,0),
                        SOAD_TAU_P = (5,255),
                        SOAD_W_N = (0,0)
                        )
            weight_spec = dict(
                        SYAM_W0_P = (5, 26),#(5, 28),
                        SYAM_W1_P = (3, 74),# Wee 
                        SYAM_W2_P = (5, 102),#(4, 150),# Wei 
                        SYAM_W3_P = (0, 0)
                        )
        elif neuron_type == "PV": # set for core 1
            neuron_spec = dict(
                        SOIF_LEAK_N = (2,10), # corrosponding to 20 ms time constant (2,5)
                        SOIF_GAIN_N = (3,60),
                        SOIF_REFR_N = (2,85),
                        SOIF_DC_P =   (0,0),
                        SOIF_SPKTHR_P = (3,60),
                        )
            conn_spec = dict(
                        DEAM_ETAU_P = (2, 30),#30
                        DEAM_EGAIN_P = (2, 35),#122
                        DEGA_ITAU_P = (1, 50),
                        DEGA_IGAIN_P = (1, 70),
                        DENM_ETAU_P = (1, 60),
                        DENM_EGAIN_P = (1, 130),
                        DENM_NMREV_N = NMDA_theta,
                        DESC_ITAU_P = (2, 16),# GAGA_A
                        DESC_IGAIN_P = (1, 90),# GAGA_A
                        SYPD_EXT_N  = (3, 200),
                        SYPD_DLY0_P = (0, 0), # Base delay 
                        SYPD_DLY1_P = (5, 250), # Precise delay 
                        SYPD_DLY2_P = mismatch_delay  # Mismatch delay
                        )
            adaptation_spec = dict(
                        SOAD_PWTAU_N = (5, 255),
                        SOAD_GAIN_P = (0,0),
                        SOAD_TAU_P = (5,255),
                        SOAD_W_N = (0,0)
                        )
            weight_spec = dict(
                        SYAM_W0_P = (0, 0), #if input is connected, otherwise not in use
                        SYAM_W1_P = (4, 31), #Wie
                        SYAM_W2_P = (4, 42), #Wii
                        SYAM_W3_P = (0, 0)
                        )
        # elif neuron_type=="SST" or neuron_type=="VIP": #IMPORTANT: Add adaptation
        #     neuron_spec = dict(
        #                 SOIF_LEAK_N = (2,5), # corrosponding to 20 ms time constant
        #                 SOIF_GAIN_N = (2,150),
        #                 SOIF_REFR_N = (2,100),
        #                 SOIF_DC_P = (0, 0),
        #                 SOIF_SPKTHR_P = (2,150),
        #                 )
        #     conn_spec = dict(
        #                 DEAM_ETAU_P = (2, 30),#30
        #                 DEAM_EGAIN_P = (2, 35),#122
        #                 DEGA_ITAU_P = (1, 50),
        #                 DEGA_IGAIN_P = (1, 70),
        #                 DENM_ETAU_P = (1, 60),
        #                 DENM_EGAIN_P = (1, 130),
        #                 DENM_NMREV_N = NMDA_theta,
        #                 DESC_ITAU_P = (2, 16),# GAGA_A
        #                 DESC_IGAIN_P = (1, 90),# GAGA_A
        #                 SYPD_EXT_N  = (3, 200),
        #                 SYPD_DLY0_P = (0, 0), # Base delay 
        #                 SYPD_DLY1_P = (5, 250), # Precise delay 
        #                 SYPD_DLY2_P = mismatch_delay  # Mismatch delay
        #                 )
        #     adaptation_spec = dict(
        #                 SOAD_PWTAU_N = (2, 160),
        #                 SOAD_GAIN_P = (1,100),
        #                 SOAD_TAU_P = (0,20),
        #                 SOAD_W_N = (1,180)
        #                 )
        #     # adaptation_spec = dict(
        #     #             SOAD_PWTAU_N = (5, 255),
        #     #             SOAD_GAIN_P = (0,0),
        #     #             SOAD_TAU_P = (5,255),
        #     #             SOAD_W_N = (0,0)
        #     #             )
        #     weight_spec = dict(
        #                 SYAM_W0_P = (0,0), # Thal_relay
        #                 SYAM_W1_P = (3, 160), # SST<-Pyr
        #                 SYAM_W2_P = (2, 100), # SST<-PV
        #                 SYAM_W3_P = (0, 0)
        #                 )
        module_spec = {}
        for spec in [neuron_spec, conn_spec,weight_spec]:
            module_spec.update(spec)
        return module_spec