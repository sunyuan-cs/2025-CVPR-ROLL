import torch.nn as nn
import torch
import torch.nn.functional as F




class SUREfcReutersNC(nn.Module):
    def __init__(self):
        super(SUREfcReutersNC, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
             
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 10))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 10))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1

class SUREfcReuters(nn.Module):
    def __init__(self):
        super(SUREfcReuters, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
             
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(10, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
             
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )
        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 10))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 10))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


class SUREfcMNISTUSPS(nn.Module):
    def __init__(self):
        super(SUREfcMNISTUSPS, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),   
            nn.Dropout(0.1),  

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),    
            nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2*num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(2*num_fea, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 256))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1




    
class SUREfcScene(nn.Module):  # 20, 59
    def __init__(self):
        super(SUREfcScene, self).__init__()
        num_fea = 512

        self.encoder0 = nn.Sequential(
            nn.Linear(20, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(59, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), nn.Dropout(0.1), 
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 20))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), nn.Dropout(0.1), 
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 59))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1






class SUREfccub(nn.Module):  # 1024, 300
    def __init__(self):
        super(SUREfccub, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Dropout(0.1),

            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Dropout(0.1),
            
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            # nn.Dropout(0.1),
           
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 1024))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 300))
    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1



class SUREfcnuswidedeep(nn.Module):  # 4096,300
    def __init__(self):
        super(SUREfcnuswidedeep, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
             
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 300))
                         
    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1
    
class SUREfcxmediadeep(nn.Module):  # 4096,300
    def __init__(self):
        super(SUREfcxmediadeep, self).__init__()
        num_fea = 512
        self.encoder0 = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(300, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),                                  
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 4096))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 300))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0)
        h1 = self.encoder1(x1)
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1



# class SUREfcxrmb(nn.Module):  # 273,112
#     def __init__(self):
#         super(SUREfcxrmb, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(273, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(112, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1), 
#                                       nn.Linear(1024, 273))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 112))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0)
#         h1 = self.encoder1(x1)
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)
#         return h0, h1, z0, z1

# class SUREfcesp(nn.Module):
#     def __init__(self):
#         super(SUREfcesp, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(100, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             # 
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # 
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(100, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             # 
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # 
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )
#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 100))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 100))


#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0.view(x0.size()[0], -1))
#         h1 = self.encoder1(x1.view(x1.size()[0], -1))
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)
#         return h0, h1, z0, z1




# class SUREfcDeepCaltech(nn.Module):
#     def __init__(self):
#         super(SUREfcDeepCaltech, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
            
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
            
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
            
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
            
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )
#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), 
#                                     #   nn.Linear(1024, 1024), nn.ReLU(),
#                                       nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 4096))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(), 
#                                     #   nn.Linear(1024, 1024), nn.ReLU(),
#                                       nn.Linear(1024, 1024), nn.ReLU(), 
#                                       nn.Linear(1024, 4096))


#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0.view(x0.size()[0], -1))
#         h1 = self.encoder0(x1.view(x1.size()[0], -1))
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder0(union)
#         return h0, h1, z0, z1




# class SUREfcDeepAnimal(nn.Module):
#     def __init__(self):
#         super(SUREfcDeepAnimal, self).__init__()
#         num_fea = 512  
#         self.encoder0 = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True),
#             nn.Dropout(0.1),

#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),   
#             nn.Dropout(0.1),         
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(4096, 512),
#             nn.BatchNorm1d(512),
#             nn.ReLU(True),
#             nn.Dropout(0.1),

#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),    
#             nn.Dropout(0.1),        
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )
#         self.decoder0 = nn.Sequential(nn.Linear(num_fea * 2, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(512, 4096))
#         self.decoder1 = nn.Sequential(nn.Linear(num_fea * 2, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(512, 4096))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0.view(x0.size()[0], -1))
#         h1 = self.encoder0(x1.view(x1.size()[0], -1))
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder0(union)
#         return h0, h1, z0, z1
    

class SUREfcNoisyMNIST(nn.Module):
    def __init__(self):
        super(SUREfcNoisyMNIST, self).__init__()
        num_fea = 512 
        self.encoder0 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),    
            nn.Dropout(0.1),       
            
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.encoder1 = nn.Sequential(
            nn.Linear(784, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.1),

            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),    
            nn.Dropout(0.1),      
            
            # nn.Linear(1024, 1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(True),
            # nn.Dropout(0.1),
            
            nn.Linear(1024, num_fea),
            nn.BatchNorm1d(num_fea),
            nn.ReLU(True)
        )

        self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                    #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 784))
        self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
                                    #   nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1),
                                      nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
                                      nn.Linear(1024, 784))

    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
        union = torch.cat([h0, h1], 1)
        z0 = self.decoder0(union)
        z1 = self.decoder1(union)
        return h0, h1, z0, z1


# class SUREfclanduse(nn.Module):
#     def __init__(self):
#         super(SUREfclanduse, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(59, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),

#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(40, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),

#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )
#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 59))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(),  nn.Dropout(0.1),
#                                       nn.Linear(1024, 40))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0.view(x0.size()[0], -1))
#         h1 = self.encoder1(x1.view(x1.size()[0], -1))
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)
#         return h0, h1, z0, z1


# class SUREfcCaltech(nn.Module):
#     def __init__(self):
#         super(SUREfcCaltech, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(1984, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(512, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )
#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1984))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 512))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0.view(x0.size()[0], -1))
#         h1 = self.encoder1(x1.view(x1.size()[0], -1))
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)

#         return h0, h1, z0, z1
    

# class SUREfcWikideep(nn.Module):  # 4096,300
#     def __init__(self):
#         super(SUREfcWikideep, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(4096, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             # 
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),      
#             nn.Dropout(0.1),      
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(300, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
            
#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),     
#             nn.Dropout(0.1),       
            
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                       nn.Linear(1024, 4096))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                       nn.Linear(1024, 300))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0)
#         h1 = self.encoder1(x1)
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)
#         return h0, h1, z0, z1
    

# class SUREfcWiki(nn.Module):  # 20, 59
#     def __init__(self):
#         super(SUREfcWiki, self).__init__()
#         num_fea = 512
#         self.encoder0 = nn.Sequential(
#             nn.Linear(128, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),

#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
            
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.encoder1 = nn.Sequential(
#             nn.Linear(10, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),

#             # nn.Linear(1024, 1024),
#             # nn.BatchNorm1d(1024),
#             # nn.ReLU(True),
#             # nn.Dropout(0.1),
            
#             nn.Linear(1024, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(True),
#             nn.Dropout(0.1),
#             # 
#             nn.Linear(1024, num_fea),
#             nn.BatchNorm1d(num_fea),
#             nn.ReLU(True)
#         )

#         self.decoder0 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 128))
#         self.decoder1 = nn.Sequential(nn.Linear(2 * num_fea, 1024), nn.ReLU(),nn.Dropout(0.1),
#                                     #   nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1),
#                                       nn.Linear(1024, 10))

#     def forward(self, x0, x1):
#         h0 = self.encoder0(x0)
#         h1 = self.encoder1(x1)
#         h0, h1 = F.normalize(h0, dim=1), F.normalize(h1, dim=1)
#         union = torch.cat([h0, h1], 1)
#         z0 = self.decoder0(union)
#         z1 = self.decoder1(union)
#         return h0, h1, z0, z1