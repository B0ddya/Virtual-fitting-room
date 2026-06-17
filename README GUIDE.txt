All files must be in one folder.
To understand how the project works, the following sequence of actions is described.
• The user stands in front of a video camera, where their face is photographed. 
  The photo is sent to a pre-built AI model to determine the shape of the person's face and cut it out from the photograph.
• The person's gender is then determined to select the required array of pre-prepared templates. The cut-out fragment is placed into these
  templates for the user to select. Once successfully selected, the desired patterns are sent to the plotter for printing.
• The plotter components are driven by gears attached to 28BYJ-48 stepper motors. These ensure smooth and precise
  movement of the writing tool across the work surface, which in turn positively impacts the quality of the results. 
• The plotter is controlled by a programmable Arduino Uno board, which receives g-code commands from a laptop. 
  ULN2003 drivers connect the control board to the motors. The plotter is powered by an external, stable 12-volt DC power source.
