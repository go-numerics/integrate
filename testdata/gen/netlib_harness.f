c Oracle harness: the netlib QUADPACK Fortran, compiled with gfortran, on the two
c fixture integrals where SciPy and this port differ (see scipyDeviations in
c quad_fixtures_test.go). Build it next to the netlib sources:
c   for f in dqawoe dqc25f dqk15w dqwgtf dqcheb dqelg dqpsrt dqawce dqc25c dqwgtc;
c   do curl -O https://www.netlib.org/quadpack/$f.f; done
c   curl -O https://www.netlib.org/linpack/dgtsl.f
c   gfortran -std=legacy -o harness netlib_harness.f dq*.f dgtsl.f d1mach.f
c with d1mach.f returning the IEEE double constants.
      program harness
      double precision fs, fq, res, abserr, alist(500), blist(500)
      double precision rlist(500), elist(500), chebmo(50,25)
      integer ier, neval, last, iord(500), nnlog(500), momcom
      external fs, fq, fi, fw
      common /cnt/ ncall
      integer ncall
      ncall = 0
      call dqawoe(fs,0.0d0,1.0d0,200.0d0,1,1.49d-8,1.49d-8,50,1,2,
     *  res,abserr,neval,ier,last,alist,blist,rlist,elist,iord,nnlog,
     *  momcom,chebmo)
      write(*,'(a,es25.17,es12.4,3i6)') 'qawo maxp1=2 ', res, abserr,
     *  neval, ier, ncall
      ncall = 0
      call dqawce(fq,-1.0d0,1.0d0,0.2d0,0.0d0,1.2d-14,200,res,abserr,
     *  neval,ier,alist,blist,rlist,elist,iord,last)
      write(*,'(a,es25.17,es12.4,4i6)') 'qawc tiny ', res, abserr,
     *  neval, ier, last, ncall
      call dqawoe(fi,0.0d0,1.0d0,30.0d0,1,1.49d-8,1.49d-8,200,1,50,
     *  res,abserr,neval,ier,last,alist,blist,rlist,elist,iord,nnlog,
     *  momcom,chebmo)
      write(*,'(a,es25.17,es12.4,3i6)') 'qawo 1/x ', res, abserr,
     *  neval, ier, last
      call dqawoe(fw,0.0d0,1.0d0,1.0d0,1,1.49d-8,1.49d-8,200,1,50,
     *  res,abserr,neval,ier,last,alist,blist,rlist,elist,iord,nnlog,
     *  momcom,chebmo)
      write(*,'(a,es25.17,es12.4,3i6)') 'qawo |x-.3|^-1', res, abserr,
     *  neval, ier, last
      end
      double precision function fw(x)
      double precision x
      fw = 0.0d0
      if (x.ne.0.3d0) fw = dabs(x-0.3d0)**(-1.0d0)
      return
      end
      double precision function fi(x)
      double precision x
      fi = 0.0d0
      if (x.gt.0.0d0) fi = 1.0d0/x
      return
      end
      double precision function fs(x)
      double precision x
      common /cnt/ ncall
      integer ncall
      ncall = ncall + 1
      fs = 0.0d0
      if (x.gt.0.0d0) fs = 1.0d0/dsqrt(x)
      return
      end
      double precision function fq(x)
      double precision x
      common /cnt/ ncall
      integer ncall
      ncall = ncall + 1
      fq = dsqrt(x+1.0d0)
      return
      end
