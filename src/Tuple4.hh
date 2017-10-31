#ifndef TUPLE4_HH
#define TUPLE4_HH

class Tuple4
{
 public:
   Tuple4(){};
   Tuple4(int ix, int iy, int iz, int ib) : ix_(ix), iy_(iy), iz_(iz), ib_(ib){}

   const int& x() const {return ix_;}
   const int& y() const {return iy_;}
   const int& z() const {return iz_;}
   const int& b() const {return ib_;}

   int& x() {return ix_;}
   int& y() {return iy_;}
   int& z() {return iz_;}
   int& b() {return ib_;}

   Tuple4& operator-=(const Tuple4& a);
   Tuple4& operator+=(const Tuple4& a);
   bool operator<(const Tuple4& b) const;

 private:
   int ix_;
   int iy_;
   int iz_;
   int ib_;
};

inline Tuple4& Tuple4::operator-=(const Tuple4& a)
{
   ix_ -= a.ix_;
   iy_ -= a.iy_;
   iz_ -= a.iz_;
   ib_ -= a.ib_;
   return *this;
}

inline Tuple4& Tuple4::operator+=(const Tuple4& a)
{
   ix_ += a.ix_;
   iy_ += a.iy_;
   iz_ += a.iz_;
   ib_ += a.ib_;
   return *this;
}

inline bool Tuple4::operator<(const Tuple4& b) const
{
   return
      ix_<b.ix_ ||
      (ix_==b.ix_ &&
       (iy_<b.iy_ ||
        (iy_ ==b.iy_ &&
         (iz_<b.iz_ ||
          (iz_==b.iz_ && ib_<b.ib_ )))));
}


inline Tuple4 operator-(const Tuple4& a, const Tuple4& b)
{
   Tuple4 c(a);
   return c-=b;
}

inline Tuple4 operator+(const Tuple4& a, const Tuple4& b)
{
   Tuple4 c(a);
   return c+=b;
}

inline bool operator==(const Tuple4& a, const Tuple4& b)
{
   return (a.x() == b.x() && a.y() == b.y() &&
           a.z() == b.z() && a.b() == b.b() );
}

inline int dot(const Tuple4& a, const Tuple4& b)
{
   return a.x()*b.x() + a.y()*b.y() + a.z()*b.z() + a.b()*b.b();
}

#endif
