// Copyright Epic Games, Inc. All Rights Reserved.

#include "Gen_AI_ExperimentGameMode.h"
#include "Gen_AI_ExperimentPlayerController.h"

AGen_AI_ExperimentGameMode::AGen_AI_ExperimentGameMode()
{
	PlayerControllerClass = AGen_AI_ExperimentPlayerController::StaticClass();
}
