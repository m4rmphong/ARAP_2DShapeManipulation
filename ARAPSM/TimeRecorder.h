#ifndef TIMERECORDER
#define TIMERECORDER

#include <MMSystem.h>
#include <Windows.h>

#pragma comment(lib, "winmm.lib")

class TimeRecorder
{
public:
	TimeRecorder()
	{
		ResetTimer();
	}

	~TimeRecorder()
	{
		tickCount = 0;
		last_tickCount = 0;
	}

	void ResetTimer()
	{
		tickCount = last_tickCount = timeGetTime();
	}

	double PassedTime()
	{
		tickCount = timeGetTime();
		return (double)(tickCount - last_tickCount) / 1000.0f;
	}

private:
	DWORD tickCount , last_tickCount;
};

#endif