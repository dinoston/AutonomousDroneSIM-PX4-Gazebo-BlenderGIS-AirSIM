// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "HumanSensorAutoRegistrationSubsystem.generated.h"

/**
 * Automatically equips people and flocking birds with sensor target components.
 * AI 사람과 군집 새에 센서 표적 컴포넌트를 자동으로 부착합니다.
 */
UCLASS()
class AUTONOMOUSDRONE_API UHumanSensorAutoRegistrationSubsystem : public UTickableWorldSubsystem
{
	GENERATED_BODY()

public:
	virtual void Tick(float DeltaTime) override;
	virtual TStatId GetStatId() const override;
	virtual bool DoesSupportWorldType(EWorldType::Type WorldType) const override;

private:
	void RegisterSensorTargets();

	float ScanAccumulator = 0.0f;
};
