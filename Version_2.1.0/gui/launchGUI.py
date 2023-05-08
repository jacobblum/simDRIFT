import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os
import sys


"""
Useful Links:
https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter Most useful in my opinion
https://www.tutorialspoint.com/python/python_gui_programming.htm
"""

pokemon_info = [['Bulbasaur', 'Grass', '318'], ['Ivysaur', 'Grass', '405'], ['Venusaur', 'Grass', '525'], ['Charmander', 'Fire', '309'], ['Charmeleon', 'Fire', '405'], ['Charizard', 'Fire', '534'], ['Squirtle', 'Water', '314'], ['Wartortle', 'Water', '405'], ['Blastoise', 'Water', '530'], ['Caterpie', 'Bug', '195'], ['Metapod', 'Bug', '205'], ['Butterfree', 'Bug', '395'], ['Weedle', 'Bug', '195'], ['Kakuna', 'Bug', '205'], ['Beedrill', 'Bug', '395'], ['Pidgey', 'Normal', '251'], ['Pidgeotto', 'Normal', '349'], ['Pidgeot', 'Normal', '479'], ['Rattata', 'Normal', '253'], ['Raticate', 'Normal', '413'], ['Spearow', 'Normal', '262'], ['Fearow', 'Normal', '442'], ['Ekans', 'Poison', '288'], ['Arbok', 'Poison', '448'], ['Pikachu', 'Electric', '320'], ['Raichu', 'Electric', '485'], ['Sandshrew', 'Ground', '300'], ['Sandslash', 'Ground', '450'], ['Nidoran?', 'Poison', '275'], ['Nidorina', 'Poison', '365'], ['Nidoqueen', 'Poison', '505'], ['Nidoran?', 'Poison', '273'], ['Nidorino', 'Poison', '365'], ['Nidoking', 'Poison', '505'], ['Clefairy', 'Fairy', '323'], ['Clefable', 'Fairy', '483'], ['Vulpix', 'Fire', '299'], ['Ninetales', 'Fire', '505'], ['Jigglypuff', 'Normal', '270'], ['Wigglytuff', 'Normal', '435'], ['Zubat', 'Poison', '245'], ['Golbat', 'Poison', '455'], ['Oddish', 'Grass', '320'], ['Gloom', 'Grass', '395'], ['Vileplume', 'Grass', '490'], ['Paras', 'Bug', '285'], ['Parasect', 'Bug', '405'], ['Venonat', 'Bug', '305'], ['Venomoth', 'Bug', '450'], ['Diglett', 'Ground', '265'], ['Dugtrio', 'Ground', '425'], ['Meowth', 'Normal', '290'], ['Persian', 'Normal', '440'], ['Psyduck', 'Water', '320'], ['Golduck', 'Water', '500'], ['Mankey', 'Fighting', '305'], ['Primeape', 'Fighting', '455'], ['Growlithe', 'Fire', '350'], ['Arcanine', 'Fire', '555'], ['Poliwag', 'Water', '300'], ['Poliwhirl', 'Water', '385'], ['Poliwrath', 'Water', '510'], ['Abra', 'Psychic', '310'], ['Kadabra', 'Psychic', '400'], ['Alakazam', 'Psychic', '500'], ['Machop', 'Fighting', '305'], ['Machoke', 'Fighting', '405'], ['Machamp', 'Fighting', '505'], ['Bellsprout', 'Grass', '300'], ['Weepinbell', 'Grass', '390'], ['Victreebel', 'Grass', '490'], ['Tentacool', 'Water', '335'], ['Tentacruel', 'Water', '515'], ['Geodude', 'Rock', '300'], ['Graveler', 'Rock', '390'], ['Golem', 'Rock', '495'], ['Ponyta', 'Fire', '410'], ['Rapidash', 'Fire', '500'], ['Slowpoke', 'Water', '315'], ['Slowbro', 'Water', '490'], ['Magnemite', 'Electric', '325'], ['Magneton', 'Electric', '465'], ["Farfetch'd", 'Normal', '377'], ['Doduo', 'Normal', '310'], ['Dodrio', 'Normal', '470'], ['Seel', 'Water', '325'], ['Dewgong', 'Water', '475'], ['Grimer', 'Poison', '325'], ['Muk', 'Poison', '500'], ['Shellder', 'Water', '305'], ['Cloyster', 'Water', '525'], ['Gastly', 'Ghost', '310'], ['Haunter', 'Ghost', '405'], ['Gengar', 'Ghost', '500'], ['Onix', 'Rock', '385'], ['Drowzee', 'Psychic', '328'], ['Hypno', 'Psychic', '483'], ['Krabby', 'Water', '325'], ['Kingler', 'Water', '475'], ['Voltorb', 'Electric', '330'], ['Electrode', 'Electric', '490'], ['Exeggcute', 'Grass', '325'], ['Exeggutor', 'Grass', '530'], ['Cubone', 'Ground', '320'], ['Marowak', 'Ground', '425'], ['Hitmonlee', 'Fighting', '455'], ['Hitmonchan', 'Fighting', '455'], ['Lickitung', 'Normal', '385'], ['Koffing', 'Poison', '340'], ['Weezing', 'Poison', '490'], ['Rhyhorn', 'Ground', '345'], ['Rhydon', 'Ground', '485'], ['Chansey', 'Normal', '450'], ['Tangela', 'Grass', '435'], ['Kangaskhan', 'Normal', '490'], ['Horsea', 'Water', '295'], ['Seadra', 'Water', '440'], ['Goldeen', 'Water', '320'], ['Seaking', 'Water', '450'], ['Staryu', 'Water', '340'], ['Starmie', 'Water', '520'], ['Scyther', 'Bug', '500'], ['Jynx', 'Ice', '455'], ['Electabuzz', 'Electric', '490'], ['Magmar', 'Fire', '495'], ['Pinsir', 'Bug', '500'], ['Tauros', 'Normal', '490'], ['Magikarp', 'Water', '200'], ['Gyarados', 'Water', '540'], ['Lapras', 'Water', '535'], ['Ditto', 'Normal', '288'], ['Eevee', 'Normal', '325'], ['Vaporeon', 'Water', '525'], ['Jolteon', 'Electric', '525'], ['Flareon', 'Fire', '525'], ['Porygon', 'Normal', '395'], ['Omanyte', 'Rock', '355'], ['Omastar', 'Rock', '495'], ['Kabuto', 'Rock', '355'], ['Kabutops', 'Rock', '495'], ['Aerodactyl', 'Rock', '515'], ['Snorlax', 'Normal', '540'], ['Articuno', 'Ice', '580'], ['Zapdos', 'Electric', '580'], ['Moltres', 'Fire', '580'], ['Dratini', 'Dragon', '300'], ['Dragonair', 'Dragon', '420'], ['Dragonite', 'Dragon', '600'], ['Mewtwo', 'Psychic', '680'], ['Mew', 'Psychic', '600']]
frame_styles = {"relief": "groove",
                "bd": 3, "bg": "#D3D3D3",
                "fg": "#B30791", "font": ("Arial", 10, "normal")}


class SetParameterScreen(tk.Tk):

    def __init__(self, *args, **kwargs):

        tk.Tk.__init__(self, *args, **kwargs)

        title_norm = {"font": ("Helvetica", 12),"bg": "#FFFFFF"}
                       
        text_norm = {"font": ("Helvetica", 10),"bg": "#D3D3D3"}

        bold_styles = {"font": ("Helvetica", 10),"bg": "#D3D3D3"}

        ital_styles = {"font": ("Helvetica", 10),"bg": "#D3D3D3"}

        main_frame = tk.Frame(self, bg="#FFFFFF", height=425, width=588)  # this is the background
        main_frame.pack(fill="both", expand="true")

        self.geometry("588x425")  # Sets window size to 500w x 300h pixels
        self.resizable(0, 0)  # This prevents any resizing of the screen

        ### Voxel Setup Entries

        frame_voxelSetup = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        frame_voxelSetup.place(y=12.5, x=12.5, height=150, width=275)

        label_title = tk.Label(frame_voxelSetup, title_norm, text="Voxel Setup")
        label_title.grid(row=0, column=0, columnspan=3, sticky='n', pady = 5)

        label_spins = tk.Label(frame_voxelSetup, text_norm, text="Number of Spins:")
        label_spins.grid(row=1, column=0, sticky='se', padx = 2, pady = 2)

        label_size = tk.Label(frame_voxelSetup, text_norm, text="Voxel Size ("+ u'\u00B5' + "m):")
        label_size.grid(row=2, column=0, sticky='se', padx = 2, pady = 2)

        label_fiberGeom = tk.Label(frame_voxelSetup, text_norm, text="Fiber Geometry:")
        label_fiberGeom.grid(row=3, column=0, sticky='se', padx = 2, pady = 2)

        label_voidDist = tk.Label(frame_voxelSetup, text_norm, text="Void Size ("+ u'\u00B5' + "m):")
        label_voidDist.grid(row=4, column=0, sticky='se', padx = 2, pady = 2)
        
        numSpins = tk.StringVar()
        numSpins.set("")
        entry_spins = ttk.Entry(frame_voxelSetup, width=22, cursor="xterm", textvariable=numSpins)
        entry_spins.grid(row=1, column=1, columnspan=2, sticky='sw', padx = 2, pady = 2)

        voxSize = tk.StringVar()
        voxSize.set("")
        entry_size = ttk.Entry(frame_voxelSetup, width=22, cursor="xterm", textvariable=voxSize)
        entry_size.grid(row=2, column=1, columnspan=2, sticky='sw', padx = 2, pady = 2)
        
        voidDist = tk.StringVar()
        voidDist.set("")
        entry_voidDist = ttk.Entry(frame_voxelSetup, width=22, cursor="xterm", textvariable=voidDist)
        entry_voidDist.grid(row=4, column=1, columnspan=2,  sticky='sw', padx = 2, pady = 2)

        fibGeom = tk.StringVar()
        fibGeom.set("")
        interwoven = ttk.Radiobutton(frame_voxelSetup, text='Interwoven', variable=fibGeom, value='Interwoven')
        interwoven.grid(row=3, column=1, sticky='sw', padx = 2, pady = 2)

        void = ttk.Radiobutton(frame_voxelSetup, text='Void', variable=fibGeom, value='Void')
        void.grid(row=3, column=2, sticky='se', padx = 2, pady = 2)


        ### Fiber Setup Entries

        frame_fiberSetup = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        frame_fiberSetup.place(y=175, x=12.5, height=175, width=275)

        label_FibTitle = tk.Label(frame_fiberSetup, title_norm, text="Fiber Setup")
        label_FibTitle.grid(row=0, column=0, columnspan=3, sticky='n', pady = 5)

        label_fib1 = tk.Label(frame_fiberSetup, text_norm, text="Fiber 1")
        label_fib1.grid(row=1, column=1,sticky='s', padx = 2, pady = 2)

        label_fib2 = tk.Label(frame_fiberSetup, text_norm, text="Fiber 2")
        label_fib2.grid(row=1, column=2,sticky='s', padx = 2, pady = 2)

        label_FFrac = tk.Label(frame_fiberSetup, text_norm, text="Vol. Fraction (0" + u'\u2013' + "1):")
        label_FFrac.grid(row=2, column=0, sticky='se', padx = 2, pady = 2)

        label_theta = tk.Label(frame_fiberSetup, text_norm, text=u'\u03B8' + ", rel. to y-axis ("+ u'\u00B0' + "):")
        label_theta.grid(row=3, column=0, sticky='se', padx = 2, pady = 2)

        label_FibRad = tk.Label(frame_fiberSetup, text_norm, text="Radius ("+ u'\u00B5' + "m):")
        label_FibRad.grid(row=4, column=0, sticky='se', padx = 2, pady = 2)

        label_fibDiff = tk.Label(frame_fiberSetup, text_norm, text="Diffusivity ("+ u'\u00B5' + "m" + u'\u00B2' + "/ms):")
        label_fibDiff.grid(row=5, column=0, sticky='se', padx = 2, pady = 2)
        
        fibFracOne = tk.StringVar()
        fibFracTwo = tk.StringVar()
        fibFracOne.set("")
        fibFracTwo.set("")

        entry_FF1 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibFracOne)
        entry_FF1.grid(row=2, column=1, sticky='sw', padx = 2, pady = 2)
        entry_FF2 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibFracTwo)
        entry_FF2.grid(row=2, column=2, sticky='sw', padx = 2, pady = 2)

        thetaOne = tk.StringVar()
        thetaTwo = tk.StringVar()
        thetaOne.set("")
        thetaTwo.set("")

        entry_theta1 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=thetaOne)
        entry_theta1.grid(row=3, column=1, sticky='sw', padx = 2, pady = 2)
        entry_theta2 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=thetaTwo)
        entry_theta2.grid(row=3, column=2, sticky='sw', padx = 2, pady = 2)
        
        fibRadOne = tk.StringVar()
        fibRadTwo = tk.StringVar()
        fibRadOne.set("")
        fibRadTwo.set("")

        entry_FR1 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibRadOne)
        entry_FR1.grid(row=4, column=1, sticky='sw', padx = 2, pady = 2)
        entry_FR2 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibRadTwo)
        entry_FR2.grid(row=4, column=2, sticky='sw', padx = 2, pady = 2)

        fibDiffOne = tk.StringVar()
        fibDiffTwo = tk.StringVar()
        fibDiffOne.set("")
        fibDiffTwo.set("")

        entry_FD1 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibDiffOne)
        entry_FD1.grid(row=5, column=1, sticky='sw', padx = 2, pady = 2)
        entry_FD2 = ttk.Entry(frame_fiberSetup, width=9, cursor="xterm", textvariable=fibDiffTwo)
        entry_FD2.grid(row=5, column=2, sticky='sw', padx = 2, pady = 2)

        ### Cell Setup Frame

        frame_cellSetup = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        frame_cellSetup.place(y=12.5, x=300, height=150, width=275)

        label_CellTitle = tk.Label(frame_cellSetup, title_norm, text="Cell Setup")
        label_CellTitle.grid(row=0, column=0, columnspan=3, sticky='n', pady = 5)

        label_cell1 = tk.Label(frame_cellSetup, text_norm, text="Cell 1")
        label_cell1.grid(row=1, column=1,sticky='s', padx = 2, pady = 2)

        label_cell2 = tk.Label(frame_cellSetup, text_norm, text="Cell 2")
        label_cell2.grid(row=1, column=2,sticky='s', padx = 2, pady = 2)

        label_CFrac = tk.Label(frame_cellSetup, text_norm, text="Vol. Fraction (0" + u'\u2013' + "1):")
        label_CFrac.grid(row=2, column=0, sticky='se', padx = 2, pady = 2)

        label_CellRad = tk.Label(frame_cellSetup, text_norm, text="Radius ("+ u'\u00B5' + "m):")
        label_CellRad.grid(row=3, column=0, sticky='se', padx = 2, pady = 2)
        
        cellFracOne = tk.StringVar()
        cellFracTwo = tk.StringVar()
        cellFracOne.set("")
        cellFracTwo.set("")

        entry_CF1 = ttk.Entry(frame_cellSetup, width=9, cursor="xterm", textvariable=cellFracOne)
        entry_CF1.grid(row=2, column=1, sticky='sw', padx = 2, pady = 2)
        entry_CF2 = ttk.Entry(frame_cellSetup, width=9, cursor="xterm", textvariable=cellFracTwo)
        entry_CF2.grid(row=2, column=2, sticky='sw', padx = 2, pady = 2)
        
        cellRadOne = tk.StringVar()
        cellRadTwo = tk.StringVar()
        cellRadOne.set("")
        cellRadTwo.set("")

        entry_CR1 = ttk.Entry(frame_cellSetup, width=9, cursor="xterm", textvariable=cellRadOne)
        entry_CR1.grid(row=3, column=1, sticky='sw', padx = 2, pady = 2)
        entry_CR2 = ttk.Entry(frame_cellSetup, width=9, cursor="xterm", textvariable=cellRadTwo)
        entry_CR2.grid(row=3, column=2, sticky='sw', padx = 2, pady = 2)


        ### Scan Setup Entries

        frame_scanSetup = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        frame_scanSetup.place(y=175, x=300, height=175, width=275)

        label_ScanTitle = tk.Label(frame_scanSetup, title_norm, text="Scan Setup")
        label_ScanTitle.grid(row=0, column=0, columnspan=3, sticky='n', pady = 5)

        label_capDelta = tk.Label(frame_scanSetup, text_norm, text="Diffusion Time, "+ u'\u0394' + " (ms):")
        label_capDelta.grid(row=1, column=0, sticky='se', padx = 2, pady = 2)

        label_smallDelta = tk.Label(frame_scanSetup, text_norm, text="Time Step, "+ u'\u1E9F' + "t (ms):")
        label_smallDelta.grid(row=2, column=0, sticky='se', padx = 2, pady = 2)

        label_diffScheme = tk.Label(frame_scanSetup, text_norm, text="Diffusion Scheme:")
        label_diffScheme.grid(row=3, column=0, sticky='e', padx = 2, pady = 2)

        capDelta = tk.StringVar()
        capDelta.set("")
        entry_capDelta = ttk.Entry(frame_scanSetup, width=15, cursor="xterm", textvariable=capDelta)
        entry_capDelta.grid(row=1, column=1, columnspan=2, sticky='sw', padx = 2, pady = 2)
        
        smallDelta = tk.StringVar()
        smallDelta.set("")
        entry_smallDelta = ttk.Entry(frame_scanSetup, width=15, cursor="xterm", textvariable=smallDelta)
        entry_smallDelta.grid(row=2, column=1, columnspan=2,  sticky='sw', padx = 2, pady = 2)

        schemeList = ["DBSI(99-Dir)", "ABCD(26-Dir)", "XYZ", "Custom..."]
        schemeVar = tk.StringVar(value=schemeList)
        entry_diffScheme = tk.Listbox(frame_scanSetup, listvariable=schemeVar,height=2,selectmode="browse",width=18)
        entry_diffScheme.grid(row=3,column=1, columnspan=2,  sticky='sw', padx = 2, pady = 2)


        """ Button Frame 1 """

        btn_frame = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        btn_frame.place(y=362.5, x=12.5, height=52.5, width=275)

        exit_button = ttk.Button(btn_frame, text="Exit", command=lambda: exit())
        exit_button.place(x=12.5, y=37.5, anchor='sw')

        default_button = ttk.Button(btn_frame, text="Defaults", command=lambda: setDefaults())
        default_button.place(x=137.5, y=37.5, anchor='s')

        clearVals_btn = ttk.Button(btn_frame, text="Clear", command=lambda: clearFields())
        clearVals_btn.place(x=262.5, y=37.5, anchor='se')

        """ Run Frame 1 """

        run_frame = tk.Frame(main_frame, bg="#D3D3D3", relief="groove", bd=2)  # This is the frame that holds the 
        run_frame.place(y=362.5, x=300, height=52.5, width=275)

        run_button = ttk.Button(run_frame, text="Start Simulation",compound="center",command=lambda: passParamsToSim())
        run_button.place(x=137.5, y=37.5, anchor='s')
            

        def setDefaults():
            numSpins.set(str(int(25e4)))
            voxSize.set(50)
            fibGeom.set("Void")
            voidDist.set(0)
            fibDiffOne.set("1.0")
            fibDiffTwo.set("2.0")
            fibRadOne.set("1.5")
            fibRadTwo.set("1.5")
            fibFracOne.set("0.3")
            fibFracTwo.set("0.3")
            thetaOne.set("0")
            thetaTwo.set("90")
            cellFracOne.set("0.1")
            cellFracTwo.set("0.1")
            cellRadOne.set("2.5")
            cellRadTwo.set("5.0")
            capDelta.set("10.0")
            smallDelta.set("0.001")
            schemeVar.set("DBSI(99-Dir)")

        def clearFields():
            numSpins.set("")
            voxSize.set("")
            fibGeom.set("")
            voidDist.set("")
            fibDiffOne.set("")
            fibDiffTwo.set("")
            fibRadOne.set("")
            fibRadTwo.set("")
            fibFracOne.set("")
            fibFracTwo.set("")
            thetaOne.set("")
            thetaTwo.set("")
            cellFracOne.set("")
            cellFracTwo.set("")
            cellRadOne.set("")
            cellRadTwo.set("")
            capDelta.set("")
            smallDelta.set("")
            schemeVar.set("")

        def passParamsToSim():
            n_spins = numSpins.get()
            vox_size= voxSize.get()
            fiberCfg= fibGeom.get()
            void_sz = voidDist.get()
            fib1_AD = fibDiffOne.get()
            fib2_AD = fibDiffTwo.get()
            fib1_R  = fibRadOne.get()
            fib2_R  = fibRadTwo.get()
            fib1_Fr = fibFracOne.get()
            fib2_Fr = fibFracTwo.get()
            fib1_Ang= thetaOne.get()
            fib2_Ang= thetaTwo.get()
            cell_F1 = cellFracOne.get()
            cell_F2 = cellFracTwo.get()
            cell_R1 = cellRadOne.get()
            cell_R2 = cellRadTwo.get()
            diffTime= capDelta.get()
            timeStep= smallDelta.get()
            dScheme = schemeVar.get()

            if ((n_spins == "") or (vox_size=="") or (fiberCfg == "") or (fib1_AD == "") or (fib1_R == "") or (fib1_Fr == "") or (fib1_Ang == "") or (cell_F1 == "") or (cell_R1 == "") or (diffTime == "") or (timeStep == "") or (dScheme == "")):
                tk.messagebox.showerror("Setup Error", "Crucial parameters are missing!\nFill in the missing parameters and press ""Start Simulation"" again.",parent=main_frame,icon=ERROR)
            if (float(fib1_Fr) + float(fib2_Fr) + float(cell_F1) + float(cell_F2)) > 1.0:
                tk.messagebox.showerror("Setup Error", "The fiber fractions and cell fractions you entered sum to a value greater than one.\nCorrect the desired parameter and press ""Start Simulation"" again.",parent=main_frame,icon=ERROR)
            if (float(cell_F1) + float(cell_F2)) > 0.6:
                tk.messagebox.showerror("Setup Error", "The cell fractions you entered sum to a value greater than the theoretical limit for random sphere packing.\nPlease adjust your cell fraction parameters such that their sum is less than 0.6, then press ""Start Simulation"" again.",parent=main_frame,icon=ERROR)
            if (float(diffTime)/float(timeStep)) < 1000:
                tk.messagebox.askokcancel(title="Setup Warning", message="The values entered for diffusion time and time step will result in fewer than 1000 time steps.\nAre you sure you want to proceed?",parent=main_frame,icon=WARNING)
            if str(fiberCfg) == "Void" and str(void_sz) == "":
                print('Empty void distance field! Assuming void distance to be zero.')
                void_sz == "0.0"
            if str(fib2_AD) == "" and str(fib2_Ang) == "" and str(fib2_Fr) == "" and str(fib2_R) == "":
                print('Passing only one fiber type to simulation...')
                fib2_AD = fib1_AD
                fib2_Ang= fib1_Ang
                fib2_R  = fib1_R
                fib2_Fr = str(0.5*float(fib1_fr))
                fib1_Fr = str(0.5*float(fib1_fr))

            if str(cell_F2) == "" and str(cell_R2 == ""):
                print('Passing only one cell type to simulation...')
                cell_F2 = str(0.5*float(cell_F1))
                cell_R2 = cell_R1

            fibADs = "\"" + str(fib1_AD) + ", " + str(fib2_AD) + "\""
            fibRs  = "\"" + str(fib1_R) + ", " + str(fib2_R) + "\""
            fibAngs= "\"" + str(fib1_Ang) + ", " + str(fib2_Ang) + "\""
            fibFrs = "\"" + str(fib1_Fr) + ", " + str(fib2_Fr) + "\""

            cellFrs= "\"" + str(cell_F1) + ", " + str(cell_F2) + "\""
            cellRs = "\"" + str(cell_R1) + ", " + str(cell_R2) + "\""

            inputArgs = r"--n_walkers " + n_spins + r" --fiber_fractions " + fibFrs + r" --fiber_radii " + fibRs + r" --thetas_Y " + fibAngs + r" --fiber_diffusions " + fibADs + r" --cell_fractions " + cellFrs + r" --cell_radii " + cellRs + r" --fiber_configuration " + fiberCfg + r" --Delta " + diffTime + r" --dt " + timeStep + r" --voxel_dims " + vox_size + r" --void_dist " + void_sz
            guiDir = __file__
            cliDir = guiDir.split(r'gui',1)[0] + r'cli.py'
            top.destroy()
            os.system("python " + cliDir + " simulate " + inputArgs)
            


top = SetParameterScreen()
top.title("dMRI-MCSIM (v2.1.0-dev)")

top.mainloop()


