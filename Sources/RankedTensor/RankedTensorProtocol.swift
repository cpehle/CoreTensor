//
//  RankedTensorProtocol.swift
//  RankedTensor
//
//  Copyright 2016-2017 The DLVM Team.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import CoreTensor

public protocol RankedTensorProtocol : ShapedArrayProtocol {
    associatedtype Rank : StaticRank where Rank.Shape == Shape
    var dynamicShape: TensorShape { get }
    init(shape: Rank.Shape, repeating repeatedValue: UnitType)
    init(shape: Rank.Shape, supplier: () -> UnitType)
    init<S: Sequence>(shape: Shape, units: S, vacancySupplier supplier: (() -> UnitType)?) where S.Element == UnitType
}

public extension RankedTensorProtocol where UnitType : Equatable {
    static func ==<T: RankedTensorProtocol>(lhs: Self, rhs: T) -> Bool where T.UnitType == UnitType {
        return lhs.dynamicShape == rhs.dynamicShape && lhs.units.elementsEqual(rhs.units)
    }

    func elementsEqual<T: RankedTensorProtocol>(_ other: T) -> Bool where T.UnitType == UnitType {
        return self == other
    }
}

public extension RankedTensorProtocol where Shape == Shape2D {
    var rowCount: UInt {
        return shape.0
    }

    var columnCount: UInt {
        return shape.1
    }
}
