// Copyright Epic Games, Inc. All Rights Reserved.

#include "HumanSensorAutoRegistrationSubsystem.h"

#include "BirdSensorTargetComponent.h"
#include "EngineUtils.h"
#include "GameFramework/Character.h"
#include "HumanSensorTargetComponent.h"

void UHumanSensorAutoRegistrationSubsystem::Tick(float DeltaTime)
{
	ScanAccumulator += DeltaTime;
	if (ScanAccumulator < 1.0f)
	{
		return;
	}
	ScanAccumulator = 0.0f;
	RegisterSensorTargets();
}

TStatId UHumanSensorAutoRegistrationSubsystem::GetStatId() const
{
	RETURN_QUICK_DECLARE_CYCLE_STAT(
		UHumanSensorAutoRegistrationSubsystem,
		STATGROUP_Tickables);
}

bool UHumanSensorAutoRegistrationSubsystem::DoesSupportWorldType(
	EWorldType::Type WorldType) const
{
	return WorldType == EWorldType::Game || WorldType == EWorldType::PIE;
}

void UHumanSensorAutoRegistrationSubsystem::RegisterSensorTargets()
{
	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return;
	}

	for (TActorIterator<AActor> It(World); It; ++It)
	{
		AActor* Actor = *It;
		if (Actor == nullptr)
		{
			continue;
		}

		const FString ActorName = Actor->GetName();
		const FString ClassName = Actor->GetClass()->GetName();
		const bool bRecognizedHuman = Actor->IsA<ACharacter>()
			&& (Actor->ActorHasTag(TEXT("HumanTarget"))
				|| ActorName.Contains(TEXT("AINormalPeople"), ESearchCase::IgnoreCase));
		if (bRecognizedHuman
			&& Actor->FindComponentByClass<UHumanSensorTargetComponent>() == nullptr)
		{
			UHumanSensorTargetComponent* HumanComponent = NewObject<UHumanSensorTargetComponent>(
				Actor,
				UHumanSensorTargetComponent::StaticClass(),
				TEXT("HumanSensorTarget_Auto"));
			if (HumanComponent != nullptr)
			{
				Actor->AddInstanceComponent(HumanComponent);
				HumanComponent->RegisterComponent();
			}
		}

		const bool bRecognizedBird = Actor->ActorHasTag(TEXT("BirdTarget"))
			|| ActorName.Contains(TEXT("FlockCharacter"), ESearchCase::IgnoreCase)
			|| ClassName.Contains(TEXT("FlockCharacter"), ESearchCase::IgnoreCase)
			|| ActorName.Contains(TEXT("BoidCharacter"), ESearchCase::IgnoreCase)
			|| ClassName.Contains(TEXT("BoidCharacter"), ESearchCase::IgnoreCase)
			|| ActorName.Contains(TEXT("Crow"), ESearchCase::IgnoreCase)
			|| ClassName.Contains(TEXT("Crow"), ESearchCase::IgnoreCase);
		if (bRecognizedBird
			&& Actor->FindComponentByClass<UBirdSensorTargetComponent>() == nullptr)
		{
			UBirdSensorTargetComponent* BirdComponent = NewObject<UBirdSensorTargetComponent>(
				Actor,
				UBirdSensorTargetComponent::StaticClass(),
				TEXT("BirdSensorTarget_Auto"));
			if (BirdComponent != nullptr)
			{
				Actor->AddInstanceComponent(BirdComponent);
				BirdComponent->RegisterComponent();
			}
		}
	}
}
