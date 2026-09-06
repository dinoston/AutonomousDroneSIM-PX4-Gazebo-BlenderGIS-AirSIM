// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Subsystems/WorldSubsystem.h"
#include "HumanSensorAutoRegistrationSubsystem.generated.h"

/**
 * Automatically equips migrated AI people with the sensor target component.
 * 마이그레이션한 AI 사람에게 센서 표적 컴포넌트를 자동으로 부착합니다.
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
	void RegisterHumanCharacters();

	float ScanAccumulator = 0.0f;
};
