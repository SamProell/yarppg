Yet another rPPG
================

*\* This is a work in progress! \**
-----------------------------------


**yarppg** is yet another implementation of remote photoplethysmography in
Python.  Remote photoplethysmography (rPPG) refers to the camera-based
measurement of a (blood volume) pulse signal.  It works by detecting small
changes in skin color, originating from the pulsation of blood [1]_.


Installation and usage
----------------------

In order to run this rPPG implementation, you first have to clone the
repository and make sure all requirements are installed.

.. code:: bash

   git clone https://github.com/SamProell/yarppg.git
   cd yarppg
   pip install -r requirements.txt

Now you can simply run the yarppg subfolder as a python package using the ``-m``
flag:

.. code:: bash

   python -m yarppg  # from yarppg top-level directory


Camera setup
------------
For rPPG to work as intended it might be required to tweak the camera
settings. For example auto white-balancing and auto-exposure could be
disabled, to avoid small adjustments in RGB values.
One possibility to change the camera settings is via the ``ffmpeg``
commandline tool. See the discussion here:
https://superuser.com/questions/1287366/open-webcam-settings-dialog-in-windows/1511657


.. [1] W Verkruysse, L O Svaasand and J S Nelson. Remote plethysmographic
   imaging using ambient light. *Optics Express*. 2008;16(26):21434–21445.
   doi:`10.1364/oe.16.021434 <https://doi.org/10.1364/oe.16.021434>`_
