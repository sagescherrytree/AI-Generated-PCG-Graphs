// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class Gen_AI_Experiment : ModuleRules
{
	public Gen_AI_Experiment(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "EnhancedInput", "ChaosVehicles", "PhysicsCore" });
	}
}
