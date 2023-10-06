#ifndef MC_TIME_INFO_INCLUDE
#define MC_TIME_INFO_INCLUDE


class MC_Time_Info
{
public:
    int    cycle;
    double initial_time;
    double final_time;
    double time;
    double time_step;

    MC_Time_Info() : cycle(0), initial_time(0.0), final_time(), time(0.0), time_step(1.0) {}

};



#endif
