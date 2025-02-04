// Copyright Epic Games, Inc. All Rights Reserved.


#include "Gen_AI_ExperimentSportsWheelRear.h"

UGen_AI_ExperimentSportsWheelRear::UGen_AI_ExperimentSportsWheelRear()
{
	WheelRadius = 40.f;
	WheelWidth = 40.0f;
	FrictionForceMultiplier = 4.0f;
	SlipThreshold = 100.0f;
	SkidThreshold = 100.0f;
	MaxSteerAngle = 0.0f;
	MaxHandBrakeTorque = 6000.0f;
}