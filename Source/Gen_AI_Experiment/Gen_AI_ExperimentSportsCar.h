// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Gen_AI_ExperimentPawn.h"
#include "Gen_AI_ExperimentSportsCar.generated.h"

/**
 *  Sports car wheeled vehicle implementation
 */
UCLASS(abstract)
class GEN_AI_EXPERIMENT_API AGen_AI_ExperimentSportsCar : public AGen_AI_ExperimentPawn
{
	GENERATED_BODY()
	
public:

	AGen_AI_ExperimentSportsCar();
};
