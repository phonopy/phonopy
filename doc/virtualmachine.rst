.. _virtualmachine:

Set up Ubuntu linux on VirtualBox
-------------------------------------------

**This document is old and can be partly different for the recent virsion of VirtualBox.**

The following the steps to install Ubuntu linux on VirtualBox. This
document is prepared for the user who failed to install phonopy for
some reason. Basically the questions on this procedure are not
accepted in the phonopy mailing list.

The setup of Ubuntu linux on VirtualBox is quite easy. Ubuntu (server)
linux on VMware player can be set up similarly.

1. Install VirtualBox

   VirtualBox is an open source virtualization system. VirtualBox runs
   on Windows, Mac, Linux, etc. On Windows and Mac, the explanation
   of how to install is unnecessary. On Ubuntu linux, it can be
   installed using ``apt-get``::

      % sudo apt-get install virtualbox

2. Download Ubuntu Server image

   The Ubuntu Server image is found at
   http://www.ubuntu.com/download/server/download. Alternatively it
   may be downloaded from the mirror sites near your location. For
   example, the file name may be like ``ubuntu-14.04-desktop-i386.iso``.

3. Create a new virtual machine

   You can specify parameters, but it is also OK just clicking next,
   next, ...

   |vbox1|

   Then you can create an empty virtual machine image.   

   |vbox2|

   To install ubuntu server, set the install image as the virtual CD
   device from Settings -> Storage, and click 'OK'.
   
   |vbox3|

   Start the virtual machine, then the installation of Ubuntu linux
   will start.
   
   |vbox4|

   In the install process, you may just click 'continue', ...,
   'install', etc. The computer's name and user name are set as you
   like.

4. System setting of the virtual machine

   Boot the virtual machine and login to the ubuntu linux with the user
   name and password.

   The terminal emulator is opened by 'Alt' + 'Ctrl' + 'T' or from the
   top-left corner seraching 'terminal'. What has to do first is
   update the system by::

      % sudo apt-get update

   ::

      % sudo apt-get upgrade

   Some packages are to be installed for convenience::

      % sudo apt-get install openssh-server

   'vim', 'zsh', 'screen', 'aptitude' may be also useful.
   Then install phonopy following :ref:`install`.

5. Using phonopy from the host computer of the virtual machine

   Phonopy can be used from the host computer (the machine where
   VirtualBox was installed).

   First, the network device of the virtual machine has to be
   modified. If NAT is used, the port-forwarding setting is required,
   Settings -> Network -> Port forwarding, right click, Insert new
   rule, Host port -> 2222, Guest port -> 22. You can login to the
   virtual machine, e.g., by terminal::

      % ssh -l username -p 2222 localhost

   (scp can be used with ``-P 2222`` option.)

   If Bridged adapter is used, you have to know the IP address of the
   virtual machine. Login to the virtual machine and in the terminal::

      % ifconfig

   The IP-address is found after ``inet addr`` of (probably)
   eth0. Then you can login to the virtual machine by the usual manner
   with the IP address.

   If the host computer is a usual linux or Mac (with the terminal in
   X11), X-forwarding is easily used by::

      % ssh -X -l username -p 2222 localhost

   or::

      % ssh -X IPADDRESS_OF_VIRTUALMACHINE

   This is very useful because the plot can be forwarded to the host
   computer.

      

.. |vbox1| image:: virtualbox-new.png
           :scale: 50

.. |vbox2| image:: virtualbox-imagenew.png
           :scale: 50

.. |vbox3| image:: vitualbox-fromimage.png 
           :scale: 50

.. |vbox4| image:: Ubuntu-install.png
           :scale: 50
