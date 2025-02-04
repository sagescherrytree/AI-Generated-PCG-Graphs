// Copyright Epic Games, Inc. All Rights Reserved.

#include "Gen_AI_ExperimentWheelFront.h"
#include "UObject/ConstructorHelpers.h"

UGen_AI_ExperimentWheelFront::UGen_AI_ExperimentWheelFront()
{
	AxleType = EAxleType::Front;
	bAffectedBySteering = true;
	MaxSteerAngle = 40.f;
}