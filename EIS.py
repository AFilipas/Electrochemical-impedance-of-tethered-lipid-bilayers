import numpy as np
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import csv
import string
import matplotlib 
from matplotlib.gridspec import GridSpec
matplotlib.rc('xtick', labelsize=12) 
matplotlib.rc('ytick', labelsize=12)
plt.rcParams['font.size'] = '12'

def Zeis012(a):
    #r'/home/crudon/Desktop/Sauliaus straipsniui/20190301/20190301 wc14(30%) meoh ph4(3) dopc-chol(50%)z.txt'
    with open(a, newline='') as inf:
        reader = csv.reader(inf, delimiter="\t")
        Frequancy=[0]
        Zreal=[0]
        Zimag=[0]
        b=0
        for i in reader:
            a=True
            b=b+1
            try:
                float(i[0])
            except ValueError:
                a=False
            if a==True:
                Frequancy=np.vstack((Frequancy,float(i[0])))
                Zreal=np.vstack((Zreal,float(i[1])))
                Zimag=np.vstack((Zimag,float(i[2])))
        Frequancy=np.delete(Frequancy,0,0)
        Zreal=np.delete(Zreal,0,0)
        Zimag=np.delete(Zimag,0,0)
        dictionary={"f":Frequancy,"Z":Zreal+1j*Zimag}
#         data=[Frequancy,Zreal+1j*Zimag]
    return dictionary
def TBLM_circuit(f, pars):
    omega=2*np.pi*f
    Rohm=pars[0]
    CPE_mem=1/(pars[1]*(omega*1j)**pars[4])
    CPE_def=1/(pars[2]*(omega*1j)**pars[5])
    Rdef=pars[3]
    C_parasitic=1/(omega*pars[6]*1j)
    Ztot=(1/Rohm+1/C_parasitic)**-1   +   (1/CPE_mem+1/(Rdef+CPE_def))**-1
    return Ztot


def Objective_Function(Circuit):
    def Objective(x0,*arg):
        Z,f=arg
        return (1/len(f))*np.sum( ((np.real(Circuit(f,x0))-np.real(Z))/np.abs(Z))**2+
                ((np.imag(Circuit(f,x0))-np.imag(Z))/np.abs(Z))**2)
    return Objective


class EIS_tblm:
    def __init__(self,Initial_guess=False,Circuit=TBLM_circuit,bounds=False,path=False,Area=0.32):
        """
        -Initial_guess is a list of parameters that is used to generate spectra if path for data file is not provided.  
        It is not used to fit the spectra. Only bounds are used to initialize fitting procedure.
        
        -Curcuit is a python function that take input in a form (frequncy, parameters). Frequancy is a list or numpy 
        array of frequancies, parameters is list or numpy array of curcuit parameters. It is used to generate objective 
        function compatable with scipy.differential_evolution. To see how to define custom function see function TBLM_curcuit:
        
        def TBLM_curcuit(f, pars):
            omega=2*np.pi*f    # numpy array of angular frequncies
            Rohm=pars[0]       
            CPE_mem=1/(pars[1]*(omega*1j)**pars[4])
            CPE_def=1/(pars[2]*(omega*1j)**pars[5])
            Rdef=pars[3]
            C_parasitic=1/(omega*pars[6]*1j)
            Ztot=(1/Rohm+1/C_parasitic)**-1   +   (1/CPE_mem+1/(Rdef+CPE_def))**-1    #equation for complex impedance 
            return Ztot       # returns numpy array of complex impedance
        
        
        
        -path is a path to csv data file you want to fit. Colums are: ......... If path is set to False spectra is automaticaly
        generated base on Initial_guess values.
        
        -Area is geometric or real area wich is used to normalize impedance per unit area.
        
        For fitting tblm spectra all input values can be left at their defoults except for Area and path. 
        
        If you want to fit custom curcuit all inputs: Intail_guess, Curcuit, bounds, and Area shuold be changed. 
        
        """
        self.fitted=False
        self.default_guess=np.array([50,5.27303933e-07,3.66057165e-06,3.75643390e+03,0.95,8.20501149e-01,3.53851916e-10])
        self.default_bounds=np.transpose([[10,2*10**-7,2*10**-6,10,0.8,0.5,10**-14],[300,10**-6,2*10**-5,100000,1,1,10**-9]])
        self.objective_function=Objective_Function(Circuit)
        self.Circuit=Circuit
        if type(bounds)!=np.ndarray:
            self.bounds=self.default_bounds
        else:
            self.bounds=bounds
        
        if Initial_guess!=np.ndarray:
            self.guess=self.default_guess
        else:
            self.guess=Initial_guess
            
        if path==False:
            print("No file provided, generated defoult spectra")
            self.f_experimental=np.logspace(-2,6,100)
            self.Z_experimental=TBLM_curcuit(self.f_experimental,self.guess)
        else:
            data=Zeis012(path)
            self.f_experimental=data["f"]
            self.Z_experimental=Area*data["Z"]

       # if bounds == False:
        #    self.bounds=self.default_bounds
            
        self.Z_guess=Circuit(self.f_experimental,self.guess)  
    def fit(self):
        """
        -compensate only works with default tBLM curcuit. While fitting other curcuits should be left at False.
        If set to False it does nothing, if set to True it removes ohmic
        resistance together with parasitic capacitence. If set to anything else removes only effects of parasytic capacitance. 
        
        """
        args=[self.Z_experimental,self.f_experimental]
        self.solution=differential_evolution(self.objective_function,self.bounds,args=args,maxiter=1000,popsize=50,polish=True,workers=1)
        self.Z_fitted=self.Circuit(self.f_experimental,self.solution.x)
        self.fitted=True
        
            
    def plot(self,axs,arg,compensate=False,**kwargs):
        if compensate ==False:
            self.CZ_fitted=self.Z_fitted
        elif compensate==True and self.fitted==True:
            self.Z_compensation=(1/(self.solution.x[0])+(np.pi*2*self.f_experimental*self.solution.x[6]*1j))**-1
            self.Ohmic_resistance=self.solution.x[0]
            self.CZ_fitted=self.Z_fitted-self.Z_compensation
        else:
            self.Z_compensation=(1/(self.solution.x[0])+(np.pi*2*self.f_experimental*self.solution.x[6]*1j))**-1
            self.Ohmic_resistance=self.solution.x[0]
            self.CZ_fitted=self.Z_fitted-self.Z_compensation+self.Ohmic_resistance
        
        self.CZreal_fitted=np.real(self.CZ_fitted)
        self.CZimag_fitted=np.imag(self.CZ_fitted)
        self.CYphase_fitted=-180/np.pi*np.angle(self.CZ_fitted)
        self.CZmod_fitted=np.abs(self.CZ_fitted)
        self.CCreal_fitted=np.real(  10**6*   ((self.CZ_fitted)*self.f_experimental*2*np.pi*1j)**-1   )
        self.CCimag_fitted=np.imag(  10**6*   ((self.CZ_fitted)*self.f_experimental*2*np.pi*1j)**-1   )
        def ColeCole(self,axs,**kwargs):
            axs.plot(self.CCreal_fitted,-self.CCimag_fitted,**kwargs)
#             axs.scatter(self.CCreal_experimental[::2],-self.CCimag_experimental[::2],facecolors='none',edgecolor=color,linewidth=2,marker=marker)
            axs.set_xlabel(r'$C_{REAL}/ \mu F cm^{-2}$')
            axs.set_ylabel(r'$-C_{IMAG}/ \mu F cm^{-2}$')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)
            axs.grid(True,linestyle="--")
        def magnitude(self,axs,**kwargs):
            axs.plot(self.f_experimental,np.abs(1/self.CZ_fitted),**kwargs)
#             axs.scatter(self.f_experimental[::2],np.abs(1/self.CZ_experimental[::2]),facecolors='none',edgecolor=color,linewidth=2,marker=marker)
            axs.set_xscale('log')
            axs.set_yscale('log')
            axs.set_ylabel(r'$|Y|,\omega ^{-1}cm^{-2}$')
            axs.set_xlabel('f/Hz')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)
            axs.grid(True,linestyle="--")
        def phase(self,axs,**kwargs):
            axs.plot(self.f_experimental,-180/np.pi*np.angle(self.CZ_fitted),**kwargs)
#             axs.scatter(self.f_experimental[::2],-180/np.pi*np.angle(self.CZ_experimental[::2]),facecolors='none',linewidth=2,edgecolor=color,marker=marker)
            axs.set_xscale('log')
            axs.set_xlabel('f/Hz')
            axs.set_ylabel(r'$argY/ deg^o$')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)  
            axs.grid(True,linestyle="--")
        def Nyquist(self,axs,**kwargs): 
            axs.plot(self.CZreal_experimental,-self.CZimag_experimental,**kwargs)
#             axs.scatter(self.CZreal_fitted[::2],-self.CZimag_fitted[::2],facecolors='none',edgecolor=color,linewidth=2,marker=marker,**scatterkwargs)
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)  
            axs.set_xlabel(r'$Z_{REAL},\omega ^{-1}cm^{-2}$')
            axs.set_ylabel(r'$Z_{IMAG},\omega ^{-1}cm^{-2}$')
            axs.grid(True,linestyle="--")
        dictionary={"ColeCole":ColeCole,"magnitude":magnitude,"phase":phase,"Nyquist":Nyquist}
        if type(axs)==type(np.array([1,2])):
            for i, ax in enumerate(axs.flat):
                dictionary[arg[i]](self,ax,**kwargs)
        else:
            dictionary[arg](self,axs,**kwargs)

    def scatter(self,axs,arg,labels=["A","B","C","D","E","F"],compensate=False,**kwargs):
        if compensate ==False:
            self.CZ_experimental=self.Z_experimental
        elif compensate==True and self.fitted==True:
            self.Z_compensation=(1/(self.solution.x[0])+(np.pi*2*self.f_experimental*self.solution.x[6]*1j))**-1
            self.Ohmic_resistance=self.solution.x[0]
            self.CZ_experimental=self.Z_experimental-self.Z_compensation
        else:
            print('r')
            self.Z_compensation=(1/(self.solution.x[0])+(np.pi*2*self.f_experimental*self.solution.x[6]*1j))**-1
            self.Ohmic_resistance=self.solution.x[0]
            self.CZ_experimental=self.Z_experimental-self.Z_compensation+self.Ohmic_resistance

            
        self.CZreal_experimental=np.real(self.CZ_experimental)
        self.CZimag_experimental=np.imag(self.CZ_experimental)
        self.CYphase_experimental=-180/np.pi*np.angle(self.CZ_experimental)
        self.CZmod_experimental=np.abs(self.CZ_experimental)
        self.CCreal_experimental=np.real(  10**6*   ((self.CZ_experimental)*self.f_experimental*2*np.pi*1j)**-1   )
        self.CCimag_experimental=np.imag(  10**6*   ((self.CZ_experimental)*self.f_experimental*2*np.pi*1j)**-1   )

        def ColeCole(self,axs,**kwargs):
#             axs.plot(self.CCreal_fitted,-self.CCimag_fitted,**kwargs)
            axs.scatter(self.CCreal_experimental[::2],-self.CCimag_experimental[::2],**kwargs)
            axs.set_xlabel(r'$C_{REAL}/ \mu F cm^{-2}$')
            axs.set_ylabel(r'$-C_{IMAG}/ \mu F cm^{-2}$')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)
            axs.grid(True,linestyle="--")
        def magnitude(self,axs,**kwargs):
#             axs.plot(self.f_experimental,np.abs(1/self.CZ_fitted),**kwargs)
            axs.scatter(self.f_experimental[::2],np.abs(1/self.CZ_experimental[::2]),**kwargs)
            axs.set_xscale('log')
            axs.set_yscale('log')
            axs.set_ylabel(r'$|Y|,\Omega ^{-1}cm^{-2}$')
            axs.set_xlabel('f/Hz')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)
            axs.grid(True,linestyle="--")
        def phase(self,axs,**kwargs):
#             axs.plot(self.f_experimental,-180/np.pi*np.angle(self.CZ_fitted),**kwargs)
            axs.scatter(self.f_experimental[::2],-180/np.pi*np.angle(self.CZ_experimental[::2]),**kwargs)
            axs.set_xscale('log')
            axs.set_xlabel('f/Hz')
            axs.set_ylabel(r'$argY/ deg^o$')
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)  
            axs.grid(True,linestyle="--")
        def Nyquist(self,axs,**kwargs): 
#             axs.plot(self.CZreal_experimental,-self.CZimag_experimental,**kwargs)
            axs.scatter(self.CZreal_fitted[::2],-self.CZimag_fitted[::2],**kwargs)
            axs.tick_params(axis='both', which='minor', bottom=False, left=False)  
            axs.set_xlabel(r'$Z_{REAL},\Omega \times cm^{-2}$')
            axs.set_ylabel(r'$Z_{IMAG},\Omega \times cm^{-2}$')
            axs.grid(True,linestyle="--")
           # axs.set_xscale('log')
           # axs.set_yscale('log')
        dictionary={"ColeCole":ColeCole,"magnitude":magnitude,"phase":phase,"Nyquist":Nyquist}
        if type(axs)==type(np.array([1,2])):
            for i, ax in enumerate(axs.flat):
                dictionary[arg[i]](self,ax,**kwargs)
                ax.text(-0.05, 1.05, labels[i], transform=ax.transAxes, size=20, weight='bold')

        else:
            dictionary[arg](self,axs,**kwargs)