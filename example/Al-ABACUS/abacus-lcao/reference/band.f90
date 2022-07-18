program main
implicit none

integer nelec,nkpt,nbnd
integer lowbound,highbound
real,allocatable :: k(:,:), kk(:), ene(:,:)
integer i,j, m,l
real vbm
real b1, b2, b3

open(unit=11,file='band.yaml')
open(unit=12,file='phonon.dat')

do i=1,24
   read(11,*)
enddo

!read(11,*) nelec,nkpt,nbnd

nkpt = 84
nbnd = 3

allocate(kk(1:nkpt))
allocate(ene(nbnd,nkpt))

do j=1,nkpt
   read(11,*)
   read(11,*)
   read(11,'(11X,F13.7)') kk(j)
   read(11,*)
   do i=1,nbnd
      read(11,*)
      read(11,'(14X,F16.10)') ene(i,j)
      read(11,*)
      read(11,*)
      read(11,*)
      read(11,*)
      read(11,*)
   enddo
enddo

!do i = 1, nbnd
!   write(12,*)
   do j=1, nkpt
      write(12,'(f17.7,1000f14.6)') kk(j), ene(1:3,j) !*33.35641
   enddo
!enddo

close(11)
close(12)

end
