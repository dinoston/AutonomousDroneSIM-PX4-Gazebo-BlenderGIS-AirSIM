// Copyright Epic Games, Inc. All Rights Reserved.

#include "HumanSensorAutoRegistrationSubsystem.h"

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
	RegisterHumanCharacters();
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

void UHumanSensorAutoRegistrationSubsystem::RegisterHumanCharacters()
{
	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return;
	}

	for (TActorIterator<ACharacter> It(World); It; ++It)
	{
		ACharacter* Character = *It;
		if (Character == nullptr || Character->FindComponentByClass<UHumanSensorTargetComponent>() != nullptr)
		{
			continue;
		}

		// Existing migrated NPCs use BP_AINormalPeople_Drone. The explicit tag
		// also supports any future human Blueprint without relying on its name.
		// 현재 마이그레이션 NPC는 BP_AINormalPeople_Drone 이름을 사용합니다.
		// 향후 다른 사람 BP는 HumanTarget 태그를 지정하면 이름에 의존하지 않습니다.
		const bool bRecognizedHuman = Character->ActorHasTag(TEXT("HumanTarget"))
			|| Character->GetName().Contains(TEXT("AINormalPeople"), ESearchCase::IgnoreCase);
		if (!bRecognizedHuman)
		{
			continue;
		}

		UHumanSensorTargetComponent* SensorComponent = NewObject<UHumanSensorTargetComponent>(
			Character,
			UHumanSensorTargetComponent::StaticClass(),
			TEXT("HumanSensorTarget_Auto"));
		if (SensorComponent != nullptr)
		{
			Character->AddInstanceComponent(SensorComponent);
			SensorComponent->RegisterComponent();
		}
	}
}
