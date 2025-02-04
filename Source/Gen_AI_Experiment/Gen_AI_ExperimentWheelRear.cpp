// Copyright Epic Games, Inc. All Rights Reserved.

#include "Gen_AI_ExperimentWheelRear.h"
#include "UObject/ConstructorHelpers.h"

UGen_AI_ExperimentWheelRear::UGen_AI_ExperimentWheelRear()
{
	AxleType = EAxleType::Rear;
	bAffectedByHandbrake = true;
	bAffectedByEngine = true;
}